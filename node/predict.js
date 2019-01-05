require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs');
global.fetch = require('node-fetch');
const fs = require('fs');
var nj = require('numjs');
var zeros = require("zeros")
var savePixels = require("save-pixels")

//random dist; helper function
function randn_bm(min, max, skew) {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );

    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) num = randn_bm(min, max, skew); // resample between 0 and 1 if out of range
    num = Math.pow(num, skew); // Skew
    num *= max - min; // Stretch to fill range
    num += min; // offset to min
    return num;
}

async function predict() {
  const model = await tf.loadModel('https://raw.githubusercontent.com/98mprice/sneaker-test/master/src/generator/model.json');

  const batch_size = 64

  // Generate noise
  let noise = nj.zeros([batch_size, 1, 1, 100])
  for (var i = 0; i < batch_size; i++) {
    for (var j = 0; j < 100; j++) {
      noise.set(i, 0, 0, j, randn_bm(-5, 5, 1))
    }
  }

  let noise_tensor = tf.tensor4d(noise.tolist())
  noise_tensor.print(true)

  // Generate images
  let generated_images = model.predict(noise_tensor)

  let output_data = await generated_images.dataSync()
  let preds = Array.prototype.slice.call(output_data);

  // Save images
  let count = 0
  for (var pic_count = 0; pic_count < batch_size; pic_count++) {
    let x = zeros([256, 256, 3]);
    for (var i = 0; i < 256; i++) {
      for (var j = 0; j < 256; j++) {
        r = (preds[count] + 1)*127.5
        g = (preds[count+1] + 1)*127.5
        b = (preds[count+2] + 1)*127.5
        x.set(i, j, 0, r)
        x.set(i, j, 1, g)
        x.set(i, j, 2, b)
        count += 3
      }
    }
    let myFile = fs.createWriteStream("output/shoe" + (64 + pic_count) + ".png");
    savePixels(x, "png").pipe(myFile)
  }
}

predict()
