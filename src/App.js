import React from 'react';
import './App.css';
import Webcam from "react-webcam";
import {Link, Toolbar, Dialog, DialogTitle, IconButton, Typography, Button,  Card, Box, AppBar,  Grid, CssBaseline, Switch,  FormControlLabel, FormLabel, Radio, RadioGroup, DialogContent} from '@mui/material';
import CategoryList from "./CategoryList"
import { Network } from './NeuralNetwork';
import {height, unit_sep, use_timer, base_timer} from './constants';
import Avatar from '@mui/material/Avatar';
import logo from "./ia.png"
import sinclogo from "./sinc-logo.png"
import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet';

const IMG_SIZE = 224

console.log(logo)

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
            category_names: ["Objeto 1", "Objeto 2"],
            images: Array(2),
            train_tensors: null, // probably ineficent
            train_labels: null,
            accuracy: [0, 0],
            scores: [0, 0],
            n_samples : [0,0], // n_samples  de la clase actual durante el entrenamiento
            output_on: -1,
            listen_keys: true,
            output_ypos: [0, 0],
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

        // create async function to load model
        async function loadModel() {
            return await mobilenet.load();
        }
        // load model
        loadModel().then((mobilenet) => {this.setState({mobilenet})});
        
        let classifier = tf.sequential();
        if(net_size === 0){
            classifier.add(tf.layers.conv2d({filters: 8, kernelSize: 3, activation: 'elu', inputShape: [IMG_SIZE, IMG_SIZE, 3]}))
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'elu'}))
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.globalMaxPooling2d({dataFormat: 'channelsLast'}))
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax'}))
        }
        if(net_size === 1){
            classifier.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'elu', inputShape: [IMG_SIZE, IMG_SIZE, 3]}))
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'elu'}))
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'elu'}))
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.globalMaxPooling2d({dataFormat: 'channelsLast'}))
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax'}))
        }    
        if(net_size === 2){
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax', inputShape: 1024}))
        }
        
        classifier.compile({loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['accuracy']});

        console.log(JSON.stringify(classifier.outputs[0].shape));

        return classifier
    }

    componentDidMount() {
        // Inicializa el modelo

        
        // local
        this.setState({classifier: this.defineNet(0, 2)})
        console.log("Init!")


        // server
        //fetch(server_url, {
        //    method: "POST",
        //    //credentials: "include",
        //    cache: "no-cache",
        //    headers: new Headers({"content-type": "application/json"})
        //}).then(response => response.json()).then(json => console.log("Init!"))

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
        category_names.push("Objeto " + (category_names.length + 1))
        let n_samples = this.state.n_samples
        n_samples.push(0)
        let zeros = Array(category_names.length).fill(0)
        
        // add column to train_labels tensor 
        let train_labels = null
        if(this.state.train_labels){
            train_labels = this.state.train_labels
            let new_column = tf.zeros([train_labels.shape[0], 1])
            train_labels = tf.concat([train_labels, new_column], 1)
        }

        this.setState({category_names, classifier: this.defineNet(this.state.net_size, category_names.length), n_samples: n_samples, category: -1, output_on: -1, classifying: false, accuracy: zeros, train_labels: train_labels})
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
            var tensor = tf.browser.fromPixels(imageData).expandDims(0);
            tensor = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE])

            var output = null
            if(this.state.net_size === 2){ 
                // MobileNet preprocessing
                const features = this.state.mobilenet.infer(tensor, true)
                output = this.state.classifier.predict(features)
            
            }
            else{
                output = this.state.classifier.predict(tensor)
            }
            
            const scores = output.arraySync()[0]
            const argmax = this.argmax(scores)
            this.setState({scores: scores, category: argmax, output_on: argmax})   
        };        
        
       
        }

    trainClassifier(){
        // count number of samples per category in train_labels
        let enoughSamples = true
        for(let i = 0; i < this.state.n_samples.length; i++)
            if(this.state.n_samples[i]<2) enoughSamples = false 

        // Fit the model only if there are at least 5 samples per category and the model is not training
        if(enoughSamples && !this.state.is_training){

            console.log('Training model...')

            this.setState({is_training: true})
            this.state.classifier.fit(this.state.train_tensors, this.state.train_labels, {
                batchSize: 4,
                epochs: 20,
                shuffle: true,
                callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1} loss: ${logs.loss}`);
                }
                },
                //yieldEvery: 'never',
            }).then(() => {
                console.log('Training completed.');
                //let accuracy = [.1, .2]
                this.setState({is_training: false})
            });
            console.log('COntiue training?...')
            

        }
    }
    
    
    addPic(category){
        console.log("add pic call")

        if(this.state.output_on === -1){
            console.log("add pic valid")

            let images = this.state.images
            let train_tensors = this.state.train_tensors
            let train_labels = this.state.train_labels
            let n_samples = this.state.n_samples
            n_samples[category] += 1
            images[category] = this.webcam.getScreenshot()
            
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
                tensor = tf.image.resizeNearestNeighbor(tensor, [IMG_SIZE, IMG_SIZE]).expandDims(0)

                // if using mobilenet, save features instead of image
                if(this.state.net_size === 2){
                    tensor = this.state.mobilenet.infer(tensor, true)
                }

                let label = Array(this.state.category_names.length).fill(0)
                label[category] = 1
                // convert label to tensor of dimension [1, 2]
                label = tf.tensor(label).expandDims(0)
                
                if (!train_tensors){
                    train_tensors = tensor
                    train_labels = label}
                else{
                    train_tensors = tf.concat([train_tensors, tensor])
                    train_labels = tf.concat([train_labels, label])}

                this.setState({n_samples: n_samples,  output_on: category, images: images, 
                    train_tensors: train_tensors, train_labels: train_labels, classifying: false})
                
                
                console.log("add pic end")

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
                    <Button onClick={()=>{this.setState({about: true})}} color="inherit">Acerca de</Button>
                    </Toolbar>
                    
                </AppBar>

                <Dialog onClose={()=>{this.setState({about: false})}} open={this.state.about}>
                    <DialogTitle>KinderNet</DialogTitle>
                    <DialogContent >
                        <Typography align="justify">
                            Este es un proyecto de aplicación web desarrollado  desde el <Link href="http://www.sinc.unl.edu.ar">sinc(i)</Link> para aprender sobre redes neuronales con alumnos de primaria y secundaria. El objetivo es que 
                            los alumnos puedan entrenar su propia red neuronal para reconocer objetos que se presenten a la camara web, de una forma interactiva. 
                            Los alumnos puedan jugar y experimentar con el proceso de entrenamiento y prueba de redes neuronales, cambiando el tamaño de la red, 
                            cantidad y tipos de clases. La red es sencilla pero puede aprender a discriminar objetos con muy pocos ejemplos.
                            <br/> <br/>
                            Más detalles en el <Link href="https://github.com/lbugnon/kinderNet">repositorio del proyecto</Link>.
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
