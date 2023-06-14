// TensorFlow
import * as tf from "@tensorflow/tfjs-node";

// Server
import express from "express";
import busboy from "busboy";
import { config } from "dotenv";
config()

let faunaModel = await tf.loadLayersModel("file://fauna_model/model.json");
// const floraModel = await tf.loadLayersModel("./models/fauna/json/fauna_model/model.json");

// * Init Express
const app = express();
const PORT = process.env.PORT || 5000;
app.use(express.json());

app.post("/predict", (req, res) => {
  if (!faunaModel) {
    res.status(500).send("Model is not loaded yet!");
    return;
  }
  
  // * Create a Busboy instance
  const bb = busboy({ headers: req.headers });
  bb.on("file", (fieldname, file, filename, encoding, mimetype) => {
    const buffer = [];
    file.on("data", (data) => {
      buffer.push(data);
    });
    file.on("end", async () => {
      // * Run Object Detection
      let tensor = tf.browser.fromPixels(Buffer.concat(buffer))
              .resizeNearestNeighbor([225, 225])
              .mean(2)
              .toFloat()
              .expandDims()
              .expandDims(-1);
      
      let predictions = await faunaModel.predict(tensor).data();
      res.json(predictions);
      const image = tf.node.decodeImage(Buffer.concat(buffer));
      // const predictions = await faunaModel.predict(image);
      // res.json(predictions);
    });
  });
  req.pipe(bb);
});



app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
});