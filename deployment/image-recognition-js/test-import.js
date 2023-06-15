const {imageClassification} = require("./file-inference")
// import imageClassification from "./file-inference";

async function call() {
  const resultFlora = await imageClassification("https://png.pngtree.com/png-vector/20210526/ourmid/pngtree-sunflower-beautiful-yellow-flower-flower-png-image_3347615.png", true);
  console.log('Result: ', resultFlora)
}
call();