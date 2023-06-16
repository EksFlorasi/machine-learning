# EksFlorasi - Machine Learning
This repository contains source code and documentation for EksFlorasi's machine learning development.

## Team Members of C23-PR499
- Machine Learning: 
  - M038DSX0335 - [Jason Andrew Gunawan](https://github.com/jasandgun)
  - M038DSX0340 - [James Rafferty Lee](https://github.com/jamesrafe)
  - M172DSX1831 - [Mohammad Azri Harahap](https://github.com/azrihrp)
- Cloud Computing: 
  - C303DSY0829 - [Calista Chandra](https://github.com/CalistaC)
  - C016DSX2599 - [Sholeh Rodhi Putra Siswantoro](https://github.com/sholehrodhi09)
- Mobile Development
  - A038DSX1146 - [Frans Wijaya](https://github.com/franswjy403)

## Machine Learning Development
EksFlorasi is using two machine learning models, to classify images for Flora and Fauna. This section will explain how our team develop EksFlorasi's machine learning models.

### Datasets Used
- [Flowers Image Classification Dataset](https://www.kaggle.com/datasets/l3llff/flowers)
- [Animal Image Classification Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

### Development Environment
> We are using Python version 3.9[^1] for this project.

Clone or download this repository.

```bash
  git clone https://github.com/EksFlorasi/machine-learning.git
```

Go to the project directory.

```bash
  cd machine-learning
```

Install all required packages.

```bash
  pip install -r requirements.txt
```

Go to notebooks directory.

```bash
  cd notebooks
```

There are four notebooks[^2] of interest:
- `modeling/dataset-preparation.ipynb` to prepare our datasets
- `modeling/fauna-transfer.ipynb` to develop a fauna model
- `modeling/flowers16-transfer.ipynb` to develop a flora model
- `inference-test.ipynb` to test our models


## Machine Learning Deployment
EksFlorasi deploys its models alongside its backend server using JavaScript. This section will explain how we use TensorFlow in JavaScript on our project.

### Deployment Environment
> We are using Node.js version 16.20[^3] for this project.
 
Go to the deployment directory from the **root directory**.

```bash
  cd deployment/image-recognition-js
```

Install all required packages.

```bash
npm install
```

Run the script after providing your models[^4].

```bash
node test-import.js
```


---
[^1]: Python 3.9: https://www.python.org/downloads/release/python-390/

[^2]: Each notebook has guides explaining the code and what they do. Should there be an error, it's most likely due to file paths.

[^3]: Node.js 16.20: https://nodejs.org/download/release/v16.20.0/

[^4]: Due to size limitations, we have uploaded our models on [Google Drive](https://drive.google.com/drive/folders/1RjQkWhZVdXS_SAwH1VX8cBzk7uBDebk-?usp=sharing). After downloading the models, update the **modelPath** variable in `image-classification.js` based off of your model location.