// TensorFlow
import * as tf from "@tensorflow/tfjs-node";
import * as fs from 'fs';

const faunaModel = await tf.loadLayersModel("file://fauna_model/model.json");
const FAUNA_LABELS = fs.readFileSync("assets/fauna_labels.txt", "utf-8").split("\n");

const readImage = path => {
  const imageBuffer = fs.readFileSync(path);
  const tfimage = tf.node.decodeImage(imageBuffer);
  return tfimage;
}

const imageClassification = async path => {
  let image = readImage(path)
      .resizeNearestNeighbor([225,225])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
  const predictions = await faunaModel.predict(image).data();

  const top5 = Array.from(predictions)
      .map(function(p, i) {
        return {
          probability: p,
          className: FAUNA_LABELS[i]
        };
      }).sort(function (a, b) {
        return b.probability - a.probability;
      }).slice(0,5);

  console.log('Classification Results:', predictions);
  console.log('Top 5 Labels: ', top5)
}

imageClassification("assets/sample_image.jpg");