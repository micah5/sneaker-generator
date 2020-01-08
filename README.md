# sneaker-generator
DCGAN that generates sneakers. Built using keras (with tensorflow backend).

Also, I've now ported across the prediction function to Tensorflow.js, and you can try it out in the browser [here](https://micah5.github.io/sneaker-generator/), or via Node.js [here](node/predict.js).

![examples](https://i.imgur.com/w4kscBP.png)

Feel free to submit any pull request, big or small. I love getting them.

## How to generate sneakers
### Python
You'll need Keras installed.
```
cd python
mkdir output
python predict.py
```
64 shoes will be generated in `output/`.

### Node.js
You'll need Tensorflow installed.
```
cd node
mkdir output
npm install
npm run test
```
64 shoes will be generated in `output/`.

### Browser
Only tested on Chrome. Takes a while, since Tensorflow.js in the browser is limited to CPU only.
[Try me!](https://micah5.github.io/sneaker-generator/)
