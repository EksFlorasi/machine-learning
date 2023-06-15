// TensorFlow
import * as tf from "@tensorflow/tfjs-node";
import * as fs from 'fs';

// Performance Monitoring
import { performance } from "perf_hooks";

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
      modelPath = "file://deployment/image-recognition-js/flora_model/flower5_90/model.json"
      labelPath = "./deployment/image-recognition-js/assets/flower5_labels.txt"
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


// let startTime1 = performance.now()
// const result1 = await imageClassification("./deployment/image-recognition-js/assets/dandelion-2.jpg", true);
// console.log('Result: ', result1)
// let endTime1 = performance.now()
// console.log(`Flower 90 ${endTime1 - startTime1} milliseconds`)

// let startTime2 = performance.now()
// const result2 = await imageClassification("./deployment/image-recognition-js/assets/dandelion-2.jpg", false);
// console.log('Result: ', result2)
// let endTime2 = performance.now()
// console.log(`Flower 86 ${endTime2 - startTime2} milliseconds`)

const resultFlora = await imageClassification("./deployment/image-recognition-js/assets/flora_image.jpg", true);
console.log('Result: ', resultFlora)

const resultFauna = await imageClassification("./deployment/image-recognition-js/assets/fauna_image.jpg", false);
console.log('Result: ', resultFauna)