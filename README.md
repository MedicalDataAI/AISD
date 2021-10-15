# AIS Diagnosis using Deep Learning
Acute Ischemic Stroke Diagnosis using Deep Learning based on CT image

If you use this code in your research, consider citing:
```
@article{
  title={xxxx},
  author={xxxx},
  journal={xxxx},
  year={xxxx},
  publisher={xxxx}
}
```

## Prerequisites

- Windows 10 with Nivida 2080Ti
- Python 3.6 with dependencies listed in the `requirements.txt` file
```cmd
   pip3 install -r requirements.txt
```


## Running

1. clone the repo to local directory
```cmd
   git clone https://github.com/MedicalDataAI/AISD.git
```

2. download the weight file of the trained model into the folder of "./model_data"
```
   localtion CNN weight: 'https://drive.google.com/file/d/1YY7-YYzRNO0cBVyAehhnEHlJ7ES0xP7n/view?usp=sharing'
   classification CNN weight: 'https://drive.google.com/file/d/1b4dxZo_C8L7iOWKGDYmShU1ukMSfuLsF/view?usp=sharing'
```

4. use the AI model to diagnose the CT slice (NOTE: to replace the parameter of path with proper location of model file)

- Predict batch CT slice (NOTE: to replace the CT slice in "./sample") into "./sample_result"
```cmd
   python3 DiagnosticAI.py
```   

- Predict single yourself CT slice into "./sample_result"
```cmd
   python3 DiagnosticAI.py "yourself file path"
```

