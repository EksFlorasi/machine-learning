const { imageClassification } = require('./image-classification');

async function executeFunction() {
  const classificationResult = await imageClassification('https://png.pngtree.com/png-vector/20210526/ourmid/pngtree-sunflower-beautiful-yellow-flower-flower-png-image_3347615.png', true);
  console.log('Classification result: ', classificationResult);
}
executeFunction();
