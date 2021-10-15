# Location and diagnosis of refractory cerebral infarction
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from PIL import Image, ImageFont, ImageDraw
import SimpleITK as sitk
import os
import numpy as np
import time
import sys
import glob
import localtionCNN, classificationCNN

def fnUseForDiagnosisSingleSlice(
        inDicomFilePath,
        inLocWeight="./model_data/localtion_cnn_weight.h5",
        inLocShape=(416, 416),
        inClfWeight="./model_data/classification_cnn_weight.h5",
        inClfShape=(64, 64),
        inAnchorsPath="./model_data/AIS_anchors.txt",
        inClassesPath="./model_data/AIS_classes.txt",
        outDir="./sample_result"
):
    modelLocaltion = localtionCNN.ConstructCNNBasedOnYOLOv3(
        inWeight=inLocWeight,
        inShape=inLocShape,
        anchors_path=inAnchorsPath,
        classes_path=inClassesPath)
    modelClassification, modelClassificationFun = classificationCNN.ConstructCNNBasedOnResnet50(
        in_input_size=inClfShape,
        in_weight=inClfWeight)
    print("CNN model load success!")
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    outBoxDir = os.path.join(outDir, "LocBox")
    if not os.path.exists(outBoxDir):
        os.makedirs(outBoxDir)

    if isinstance(inDicomFilePath, str):
        inDicomFilePath = [inDicomFilePath]
    for dcmIndex, dcmFp in enumerate(inDicomFilePath):
        _, dcmBaseFileName = os.path.split(dcmFp)
        dcmFileName, dcmFileExtension = os.path.splitext(dcmBaseFileName)

        image = fnReadDcmIntoImageObj(dcmFp)
        predictBoxs = modelLocaltion.predictSingleImg(image) # [left, top, right, bottom]
        showFont = ImageFont.truetype(font='model_data/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        showBoxThickness = (image.size[0] + image.size[1]) // 300
        imageDiagnosis = image.convert('RGB')
        for boxIndex, singleBox in enumerate(predictBoxs):
            imageNp = np.array(image)
            imageCropNp = imageNp[int(singleBox[1]):int(singleBox[3]), int(singleBox[0]):int(singleBox[2]), ]
            imgForClf = Image.fromarray(imageCropNp)
            imgForClf = imgForClf.convert("L")
            imgForClfFp = os.path.join(outBoxDir, "ForClf_"+dcmFileName+"_"+str(boxIndex)+".bmp")
            imgForClf.save(imgForClfFp)
            clfScore = classificationCNN.predictSingleImg(modelClassification, imgForClfFp, inClfShape, modelClassificationFun)
            if clfScore >= 0.5:
                print(r"AIS in [%d / %d] localtion predict box in %s(Risk=%.2f)." % (boxIndex+1, len(predictBoxs), dcmFileName, clfScore))
                showBoxText = "AI_risk=%.2f" % clfScore
                drawObj = ImageDraw.Draw(imageDiagnosis)
                drawTextSize = drawObj.textsize(showBoxText, showFont)
                if singleBox[1] - drawTextSize[1] >= 0:
                    textOrigin = np.array([singleBox[0], singleBox[1] - drawTextSize[1]])
                else:
                    textOrigin = np.array([singleBox[0], singleBox[1] + 1])
                for thicknessIndex in range(showBoxThickness):
                    drawObj.rectangle(
                        [singleBox[0] + thicknessIndex, singleBox[1] + thicknessIndex, singleBox[2] - thicknessIndex, singleBox[3] - thicknessIndex],
                        outline="yellow")
                drawObj.rectangle(
                    [tuple(textOrigin), tuple(textOrigin + drawTextSize)],
                    fill="yellow")
                drawObj.text(textOrigin, showBoxText, fill=(0, 0, 0), font=showFont)
            else:
                print(r"Non-AIS in [%d / %d] localtion predict box in %s(Risk=%.2f)." % (boxIndex + 1, len(predictBoxs), dcmFileName, clfScore))
                showBoxText = "AI_risk=%.2f" % clfScore
                drawObj = ImageDraw.Draw(imageDiagnosis)
                drawTextSize = drawObj.textsize(showBoxText, showFont)
                if singleBox[1] - drawTextSize[1] >= 0:
                    textOrigin = np.array([singleBox[0], singleBox[1] - drawTextSize[1]])
                else:
                    textOrigin = np.array([singleBox[0], singleBox[1] + 1])
                for thicknessIndex in range(showBoxThickness):
                    drawObj.rectangle(
                        [singleBox[0] + thicknessIndex, singleBox[1] + thicknessIndex, singleBox[2] - thicknessIndex,
                         singleBox[3] - thicknessIndex],
                        outline="green")
                drawObj.rectangle(
                    [tuple(textOrigin), tuple(textOrigin + drawTextSize)],
                    fill="green")
                drawObj.text(textOrigin, showBoxText, fill=(0, 0, 0), font=showFont)
        if 0 == len(predictBoxs):
            print(r"Non-AIS in %s." % dcmFileName)
        imageDiagnosisSaveFp = os.path.join(outDir, "Diagnosis_"+dcmFileName+".bmp")
        imageDiagnosis.save(imageDiagnosisSaveFp)
    pass

def fnReadDcmIntoImageObj(inDcmFp, dcmWindowLevel=[30, 60], inMultiImg=4):
    reader = sitk.ImageFileReader()
    reader.SetImageIO('GDCMImageIO')
    reader.SetFileName(inDcmFp)
    reader.ReadImageInformation()

    singleDcmImg = reader.Execute()
    dcmMin = dcmWindowLevel[0] - dcmWindowLevel[1] / 2
    dcmMax = dcmWindowLevel[0] + dcmWindowLevel[1] / 2
    singleDcmImgNp = sitk.GetArrayFromImage(singleDcmImg)
    max_pixel_value = singleDcmImgNp.max()
    singleDcmImg = sitk.Threshold(singleDcmImg, dcmMin, max_pixel_value + 1.0, dcmMin)
    singleDcmImg = sitk.Threshold(singleDcmImg, dcmMin - 1, dcmMax, dcmMax)
    imageRaw = np.squeeze(sitk.GetArrayFromImage(singleDcmImg))
    image = Image.fromarray(imageRaw * inMultiImg)
    return image


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        print("Input dicom file path: %s." % sys.argv[1])
        fnUseForDiagnosisSingleSlice(sys.argv[1])
    else:
        fnUseForDiagnosisSingleSlice(glob.glob("./sample/*.dcm"))
    pass
