// TensorFlow
import * as tf from "@tensorflow/tfjs-node";
import * as fs from 'fs';


// Read image from file system
const readImage = path => {
  const imageBuffer = fs.readFileSync(path);
  return tf.node.decodeImage(imageBuffer);
}


// Classify Images
const imageClassification = async (path, isFlora) => {
  // Model and label paths (bucket address/path)
  let modelPath = "";
  let labelPath = "";
  let imageSize = [];

  // Branching to select paths and image size
  if (isFlora) {
      modelPath = "{PATH TO FLORA MODEL}"
      labelPath = "{PATH TO FLORA LABEL}"
      imageSize = [150, 150]
  } else  {
      modelPath = "file://deployment/image-recognition-js/fauna_model/model.json";
      labelPath = "./deployment/image-recognition-js/assets/fauna_labels.txt"
      imageSize = [225, 225]
  }

  // Load model and label based on chosen classification
  const chosenModel = await tf.loadLayersModel(modelPath);
  const chosenLabel = fs.readFileSync(labelPath, "utf-8").split("\r\n");

  // Get image from file system
  let image = readImage(path)
  // Preprocess the image to fit the model
      .resizeNearestNeighbor(imageSize)
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();

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

  // console.log('Classification Results:', predictions);
  // console.log('Top 5 Labels: ', top5)
  return top5;
}


const result = await imageClassification("./deployment/image-recognition-js/assets/sample_image.jpg", false);
console.log('Result: ', result)