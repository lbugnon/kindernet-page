import React from 'react';
import './App.css';
import Webcam from "react-webcam";
import {Link, Toolbar, Dialog, DialogTitle, IconButton, Typography, Button,  Card, Box, AppBar,  Grid, CssBaseline, Switch,  FormControlLabel, FormLabel, Radio, RadioGroup, DialogContent} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import CategoryList from "./CategoryList"
import { Network } from './NeuralNetwork';
import {height, unit_sep, use_timer, base_timer} from './constants';
import Avatar from '@mui/material/Avatar';
import logo from "./ia.png"
import sinclogo from "./sinc-logo.png"
import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet';

const IMG_SIZE = 64

// event listener
class EventListener extends React.Component{
    componentDidMount() {
        window.addEventListener("keyup",this.props.onKeyUp)
    }
    componentWillUnmount() {
        window.removeEventListener("keyup",this.props.onKeyUp)
    }
    render(){
        return null;
    }
}

// Kindernet ==========================================
class KinderNet extends React.Component{
    constructor(props){
        super(props);
        this.state={
            classifier: null,
            is_training: false,
            category: -1,
            classifying: false,
            net_size: 0, // mayor valor, mas compleja la red
            category_names: ["Cosa 1", "Cosa 2"],
            images: Array(2),
            train_tensors: null, // probably ineficent
            train_features: null,
            train_labels: null,
            test_tensors: null, 
            test_features: null,
            test_labels: null,
            accuracy: [0, 0],
            scores: [0, 0],
            n_samples : [0,0], // n_samples  de la clase actual durante el entrenamiento
            output_on: -1,
            listen_keys: true,
            output_ypos: [0, 0],
            help: false,
            about: false
        };
        this.response = null
        this.captureGlobalEvent = this.captureGlobalEvent.bind(this);
        this.handleTransitionEnd = this.handleTransitionEnd.bind(this);
        this.handleTimerOut = this.handleTimerOut.bind(this);
        this.captureCategoryNames = this.captureCategoryNames.bind(this);
        
        this.handleTrain = this.handleTrain.bind(this);
        this.handleClassifierChange = this.handleClassifierChange.bind(this);
        this.handleAddCategory = this.handleAddCategory.bind(this);
        this.handleRemoveCategory = this.handleRemoveCategory.bind(this);
        this.handleKeyListen = this.handleKeyListen.bind(this);
        
        this.classifyPic = this.classifyPic.bind(this);
    }
    setRef = webcam => {
        this.webcam = webcam;
    };

  
    defineNet(net_size, nclasses){
        
        let classifier = tf.sequential();
        if(net_size === 0){
            classifier.add(tf.layers.conv2d({filters: 8, kernelSize: 3, activation: 'elu', inputShape: [IMG_SIZE, IMG_SIZE, 3]}))
            classifier.add(tf.layers.batchNormalization())
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.globalAveragePooling2d({dataFormat: 'channelsLast'}))
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax'}))
        }if(net_size === 1){
            classifier.add(tf.layers.conv2d({filters: 8, kernelSize: 3, activation: 'elu', inputShape: [IMG_SIZE, IMG_SIZE, 3]}))
            classifier.add(tf.layers.batchNormalization())
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'elu'}))
            classifier.add(tf.layers.batchNormalization())
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.globalAveragePooling2d({dataFormat: 'channelsLast'}))
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax'}))
        }
        if(net_size === 2){
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax', inputShape: 1024}))
        }
        
        classifier.compile({loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['accuracy']});
        console.log(classifier.summary())
        return classifier
    }

    componentDidMount() {
        // Inicializa el modelo

        // create async function to load model
        async function loadModel() {
            return await mobilenet.load();
        }
        // load model
        loadModel().then((mobilenet) => {this.setState({mobilenet})});


        this.setState({classifier: this.defineNet(this.state.net_size, this.state.n_samples.length)})
        console.log("Init!")

        // Inicializa el timer
        if(use_timer)
            setTimeout(this.handleTimerOut, base_timer)
    }

    handleClassifierChange(net_size){
        this.setState({net_size: net_size, classifier: this.defineNet(net_size, this.state.category_names.length), category: -1, output_on: -1, classifying: false, accuracy: Array(this.state.category_names.length).fill(0)})
        
        return
    }
    handleAddCategory(){
        let category_names = this.state.category_names
        category_names.push("Cosa " + (category_names.length + 1))
        let n_samples = this.state.n_samples
        n_samples.push(0)
        let zeros = Array(category_names.length).fill(0)
        
        // add column to labels tensor 
        let train_labels = null
        if(this.state.train_labels){
            train_labels = this.state.train_labels
            let new_column = tf.zeros([train_labels.shape[0], 1])
            train_labels = tf.concat([train_labels, new_column], 1)
        }

        let test_labels = null
        if(this.state.test_labels){
            test_labels = this.state.test_labels
            let new_column = tf.zeros([test_labels.shape[0], 1])
            test_labels = tf.concat([test_labels, new_column], 1)
        }

        this.setState({category_names, classifier: this.defineNet(this.state.net_size, category_names.length), n_samples, category: -1, output_on: -1, classifying: false, accuracy: zeros, train_labels, test_labels})
        return
    }
    handleRemoveCategory(category){
        let category_names = this.state.category_names
        let images = this.state.images        
        category_names.splice(category, 1)
        images.splice(category, 1)
        
        this.serverCall("/eliminar_categoria/", {category: category}).then(
            response=>this.setState({n_samples: response.n_samples, images: images, category_names: category_names, category: -1, 
                                     output_on: -1, classifying: false, accuracy: Array(this.state.category_names.length).fill(0)}))

        return
    }
    handleTimerOut(){

        if(!this.state.classifying){
            setTimeout(this.handleTimerOut, base_timer)
            return
        }
        
        this.classifyPic()

        setTimeout(this.handleTimerOut, base_timer)
        return
    }
    handleTransitionEnd(){
        this.setState({output_on: -1})
    }
    handleTrain(category){
        this.addPic(category)
        this.trainClassifier()
    }
    handleKeyListen(is_enabled){
        this.setState({listen_keys: is_enabled})
    }


    argmax(array){ return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1]}

    classifyPic(){
        const base64ImageData =  this.webcam.getScreenshot()

        // Create a new Image object
        const image = new Image();

        // Set the source of the image as the Base64 image data
        image.src = base64ImageData;

        // Wait for the image to load
        image.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            const context = canvas.getContext('2d');
            context.drawImage(image, 0, 0);
            const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
            
            var output = null
            var scores = null
            tf.tidy(() => {
                var tensor = tf.browser.fromPixels(imageData).expandDims(0);
                tensor = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE])

                if(this.state.net_size === 2){ 
                    // MobileNet preprocessing
                    output = this.state.classifier.predict(this.state.mobilenet.infer(tensor, true))
                }
                else{
                    output = this.state.classifier.predict(tensor)
                }
                scores = output.arraySync()[0]
            }
            )
            
            const argmax = this.argmax(scores)
            this.setState({scores: scores, category: argmax, output_on: argmax})   
        };        
        
       
        }

    trainClassifier(){
        // count number of samples per category in train_labels
        console.log("32?", tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE'))
        let enoughSamples = true
        for(let i = 0; i < this.state.n_samples.length; i++)
            if(this.state.n_samples[i]<6) enoughSamples = false 

        // Fit the model only if there are at least 5 samples per category and the model is not training
        if(enoughSamples && !this.state.is_training){

            console.log('Training model...')
            this.setState({is_training: true})
         
            tf.tidy(() => {
                let train_input, test_input
                if(this.state.net_size<2){
                    train_input = this.state.train_tensors
                    test_input = this.state.test_tensors
                }
                else{
                    train_input = this.state.train_features
                    test_input = this.state.test_features
                }

                this.state.classifier.fit(train_input, this.state.train_labels, {
                    batchSize: 4,
                    epochs: 10,
                    shuffle: true,
                    validationData: [test_input, this.state.test_labels],
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch + 1} loss: ${logs.loss.toFixed(2)} acc: ${logs.acc.toFixed(2)} val_loss: ${logs.val_loss.toFixed(2)} val_acc: ${logs.val_acc.toFixed(2)}`);
                    }
                    },
                    yieldEvery: 'never',
                }).then(() => {
                    
                    // if train_input is shorter than train_labels, remove the last column of train_labels
                    //let train_labels = this.state.train_labels
                    //if(train_input.shape[0] < this.state.train_labels.shape[0]){
                    //    train_labels = train_labels.slice([0, 0], [train_input.shape[0], train_labels.shape[1]])}

                    // convert one-hot encoding to integer labels
                    let test_labels_int = []
                    for(let i = 0; i < this.state.test_labels.shape[0]; i++){
                        test_labels_int.push(this.argmax(this.state.test_labels.arraySync()[i]))
                    }
                    
                    // get accuracy per class:
                    let accuracy = Array(this.state.category_names.length).fill(0)
                    let n_samples = Array(this.state.category_names.length).fill(0)
                    let predictions = this.state.classifier.predict(test_input).arraySync()

                    for(let i = 0; i < predictions.length; i++){
                        let argmax = this.argmax(predictions[i])
                        if(argmax === test_labels_int[i]){
                            accuracy[argmax] += 1
                            n_samples[argmax] += 1
                        }
                    }
                    for(let i = 0; i < accuracy.length; i++){
                        if(n_samples[i] > 0) accuracy[i] = accuracy[i]/n_samples[i]
                    }
                    console.log('Training completed.');
                    
                    this.setState({accuracy, is_training: false})
                    
                });
                
            });
        }
    }
    
    
    addPic(category){

        if(this.state.output_on === -1){

            let images = this.state.images
            let n_samples = this.state.n_samples
            n_samples[category] += 1
            images[category] = this.webcam.getScreenshot()
            
            // the first images per class go to test
            var tensors, features, labels
            if(n_samples[category] < 5){
                tensors = this.state.test_tensors
                features = this.state.test_features
                labels = this.state.test_labels}
            else{
                tensors = this.state.train_tensors
                features = this.state.train_features
                labels = this.state.train_labels}

            const image = new Image();
            image.src = images[category];

            image.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = image.width;
                canvas.height = image.height;
                const context = canvas.getContext('2d');
                context.drawImage(image, 0, 0);
                const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
                var tensor = tf.browser.fromPixels(imageData);

                tensor = tf.image.resizeNearestNeighbor(tensor, [224, 224]).expandDims(0)
                let feature = this.state.mobilenet.infer(tensor, true)

                tensor = tf.image.resizeNearestNeighbor(tensor, [IMG_SIZE, IMG_SIZE])

                let label = Array(this.state.category_names.length).fill(0)
                label[category] = 1
                // convert label to tensor of dimension [1, 2]
                label = tf.tensor(label).expandDims(0)
                
                if (!tensors){
                    tensors = tensor
                    features = feature
                    labels = label}
                else{
                    tensors = tf.concat([tensors, tensor])
                    features = tf.concat([features, feature])
                    labels = tf.concat([labels, label])}
                
                if(n_samples[category] < 3){
                        this.setState({n_samples: n_samples,  output_on: category, images, 
                          test_tensors: tensors, test_features: features, test_labels: labels, classifying: false})
                        }
                else{
                    this.setState({n_samples: n_samples,  output_on: category, images, 
                            train_tensors: tensors, train_features: features, train_labels: labels, classifying: false})
                        }
                               

              };        
              
           
           
            }
    }

    captureGlobalEvent(e) {
        if(this.state.listen_keys){
            // entrenamiento
            if (e.key <= this.state.category_names.length) {
                const category = Number(e.key)-1
                this.addPic(category)
                this.trainClassifier()
            }
            if (e.key === "c") 
                this.setState({classifying: !this.state.classifying})
        }        
        

    }

    captureCategoryNames(i, name){
        let names = this.state.category_names
        names[i] = name
        this.setState({category_names: names}) 
    }
        

    render(){
        const videoConstraints = {
            width: 350,
            height: 350,
            facingMode: "user"
        };

        let pred_message
        if(!this.state.classifying)
            pred_message = ""
        else
            if(this.state.category!==-1)
                pred_message = "¡Es '" + this.state.category_names[this.state.category] + "'!"
           
        let ypos = []
        for (let i = 0; i <this.state.n_samples.length; i++) 
            ypos[i] = height / 2 + unit_sep[2] * (i - this.state.n_samples.length / 2)

        return(
            <Box className="noselect">

                <AppBar position="static">
                    <Toolbar>
                    <IconButton>
                    <Avatar alt="Kindernet logo" src={logo} />
                    </IconButton>
                    <Typography variant="h5" component="div" sx={{ flexGrow: 1 }}>
                        KinderNet: ¡Enseñemos a la compu a ver!
                    </Typography>
                    <Button onClick={()=>{this.setState({help: true})}} color="inherit">Ayuda</Button>
                    <Button onClick={()=>{this.setState({about: true})}} color="inherit">Acerca de</Button>
                    </Toolbar>
                    
                </AppBar>

                <Dialog onClose={()=>{this.setState({about: false})}} open={this.state.about}>
                    <DialogTitle>KinderNet</DialogTitle>
                    <DialogContent >
                        <Typography align="justify">
                            Este es un proyecto de aplicación web desarrollado  desde el <Link href="http://www.sinc.unl.edu.ar">sinc(i)</Link> para aprender sobre redes neuronales con alumnos de primaria y secundaria. El objetivo es que 
                            los alumnos puedan entrenar su propia red neuronal para reconocer cosas que se presenten a la camara web, de una forma interactiva. 
                            Los alumnos puedan jugar y experimentar con el proceso de entrenamiento y prueba de redes neuronales, cambiando el tamaño de la red, 
                            cantidad y tipos de clases. La red es sencilla pero puede aprender a discriminar cosas con muy pocos ejemplos.
                            <br/> <br/>
                            Más detalles en el <Link href="https://github.com/lbugnon/kinderNet">repositorio del proyecto</Link>.
                        </Typography>
                            
                    </DialogContent>
                </Dialog>

                <Dialog onClose={()=>{this.setState({help: false})}} open={this.state.help}>
                    <DialogTitle>Instrucciones</DialogTitle>
                    <DialogContent >
                        <Typography align="justify">
                            Antes de comenzar, definamos las cosas que vamos a clasificar. Por ejemplo: "manzana" y "banana" en lugar de "Cosa 1" y "Cosa 2".
                            
                            <br/> <br/>
                        
                            Para comenzar a entrenar la red neuronal, ubicar la primer cosa en la webcam y apretar el 1 o hacer click en la neurona correspondiente de la derecha. Esto va a tomar una foto y pasársela a la red para que vaya aprendiendo.

                            <br/> <br/>
                        
                            Hacer lo mismo con la otra clase hasta que tenga al menos 6 ejemplos cada una. Las barras de la derecha indican qué tan bien la red está aprendiendo cada clase. Si la barra está llena, la red ya aprendió todo lo que puede de esa cosa. Si la barra está vacía, la red no sabe nada de esa cosa.

                            <br/> <br/>

                            Una vez que la red haya aprendido, se puede probar haciendo click en el botón de la izquierda para que indique "Probando". La cámara tomará fotos y las pasará por la red para que indique la cosa que reconoce. 
    
                            <br/> <br/>

                            Podés sumar más cosas haciendo click en el botón <AddIcon/> a la derecha. También podés borrar una cosas haciendo click en el botón <DeleteIcon/>. Refrescando la página (F5) se borra todo y se vuelve a empezar. 

                            

                        </Typography>
                            
                    </DialogContent>
                </Dialog>

                <EventListener onKeyUp={this.captureGlobalEvent}/>
                <Grid  container pt={10} justifyContent='center' textAlign='center'>
                    
                    <Grid item pt={15} sm={4} lg={2}>
                        <Webcam videoConstraints = {videoConstraints} audio={false} ref={this.setRef} screenshotFormat="image/png" quality={1} className="Webcam"/> 
                        <Card variant="outlined">
                            <h2>Panel de control</h2>
                            <Grid container justifyContent='center' alignItems='center'>
                                <Grid style={{color:this.state.classifying? "black":"gray"}}><h3>Probando</h3></Grid>
                                <Switch checked={!this.state.classifying} onChange={()=>{this.setState({classifying: !this.state.classifying})}} />
                                <Grid style={{color:this.state.classifying? "gray":"black"}}><h3>Aprendiendo</h3></Grid>
                            </Grid> 
                            <FormLabel id="radio-buttons-size">Tamaño de la red neuronal</FormLabel>
                            <Grid container justifyContent='center' alignItems='center'>
                                <RadioGroup aria-labelledby="radio-buttons-size" defaultValue="Pequeña">
                                    <FormControlLabel value="Pequeña" control={<Radio onChange={()=>{this.handleClassifierChange(0)}}/>} 
                                    label="Pequeña" />
                                    <FormControlLabel value="Mediana" control={<Radio onChange={()=>{this.handleClassifierChange(1)}}/>} 
                                    label="Mediana" />
                                    <FormControlLabel value="Grande" control={<Radio onChange={()=>{this.handleClassifierChange(2)}}/>} 
                                    label="Grande" />
                                </RadioGroup>
                            </Grid> 
                        </Card>
                        
                    </Grid>

                    <Grid item sm={4}>
                        <Network onClick={this.handleTrain} category = {this.state.output_on} onTransitionEnd = {this.handleTransitionEnd}
                            size = {this.state.net_size} n_outputs = {this.state.category_names.length}
                            classifying = {this.state.classifying} />        
                         
                        <h1>{pred_message}</h1>   
            
                    </Grid>

                    <Grid item sm={4} lg={2}>
                        <CategoryList images = {this.state.images}  scores={this.state.classifying?this.state.scores:this.state.accuracy} ypos={ypos} enableKeys={this.handleKeyListen} category_names={this.state.category_names} n_samples={this.state.n_samples} 
            get_category_names={this.captureCategoryNames} handleAddCategory={this.handleAddCategory} handleRemoveCategory={this.handleRemoveCategory}/>     
                    </Grid>

                </Grid>
                
                <Grid  container pr={10} mt={-10} justifyContent='right' textAlign='center'>
                <a href="http://sinc.unl.edu.ar">
                    <img src={sinclogo} style={{height: 70}} alt={"sinc(i) logo"} />
                </a>
                </Grid>
            </Box>
        );
    }

}



function App() {
  return (
      <React.Fragment>
          <CssBaseline />
          <KinderNet />
      </React.Fragment>

  );
}

export default App;
