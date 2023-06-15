// TensorFlow
const tf = require("@tensorflow/tfjs-node")
const fs = require("fs");
const axios = require("axios")

// Read image from url
const readImage = async url => {
  const response = await axios.get(url,  { responseType: 'arraybuffer' })
  const buffer = Buffer.from(response.data, "utf-8")
  return buffer
}

// Crop image (to use with multiple file formats)
const cropImage = img => {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}


// Image classifier function
const imageClassification = async (path, isFlora) => {
  // Model and label paths (bucket address/path)
  let modelPath = "";
  let labelPath = "";
  let imageSize = [];

  // Branching to select paths and image size
  if (isFlora) {
      modelPath = "file://deployment/image-recognition-js/flora_model/flower16_91/model.json"
      labelPath = "./deployment/image-recognition-js/assets/flora_labels.txt"
      imageSize = [225, 225]
  } else  {
      modelPath = "file://deployment/image-recognition-js/fauna_model/model.json";
      labelPath = "./deployment/image-recognition-js/assets/fauna_labels.txt"
      imageSize = [225, 225]
  }

  // Load model and label based on chosen classification
  const chosenModel = await tf.loadLayersModel(modelPath);
  const chosenLabel = fs.readFileSync(labelPath, "utf-8").split("\r\n");

  // Get image from file system
  let imgBuf = await readImage(path)
  imgBuf = tf.node.decodeImage(imgBuf);

  // Preprocess the image to fit the model
  let image = cropImage(imgBuf)
      .resizeNearestNeighbor(imageSize)
      .expandDims(0)
      .toFloat()
      .div(tf.scalar(127))
      .sub(tf.scalar(1));

  // Predict the image
  const predictions = await chosenModel.predict(image).data();
  
  // Summarize to top 5 predictions
  const top5 = Array.from(predictions)
      .map(function(p, i) {
        return {
          probability: p,
          label: chosenLabel[i]
        };
      }).sort(function (a, b) {
        return b.probability - a.probability;
      }).slice(0,5);

  return top5;
}

module.exports = {imageClassification};
