import {Grid, Stack, CardMedia, Typography} from '@mui/material';
import React from 'react';


export default function ImagesList(props){
        
    const blank_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAAABGdBTUEAALGPC/xhBQAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAAd0SU1FB9wKHhMRLSoeQSUAAAFLSURBVGje7ZVPSgMhDEMfg/nFgIgW7gMYJWKIGQQdZCIROJFUYU2D3IggyRgwiCULI0EGE8YUdJivP/f6mSs2w2ZMzM73Pztz7zTmnmf03gPU9v/6Ii7G+0f9W3Ab4ANg6wVwA0GOMZSTYEUBPQMnAKYDOAOkClwCTAO4BlAAmAO0CVAAdACcD1wALgA1ABcD5wB0AMMA/AAoANcB2wC3AE8A6QJdAI0A7gDYAA4D8ABOAKcBbwC5AB8AZQEsADwAPwKpAM0ANwAqAPYAwQArAAkADgG9AJwA1wAkAEMAfQD5AMIA9wBfAGoA2wDaADMDtAHHAGkA1wKbAHYAEQBGADIA2gASAKIAKwCtADcA2gATANwAfgA/AJoALwA6ADgAfwAgAM0ANQCYAMoAZQD/AIQAlwBqAAwAygBOAAsA0wA1AIgA3AA7AKkALQDaAK0ACQAOALMAZQDVANcAEQD7AEsAMwCiAH8ARwAyAHoAigBGACsAqQAHAEoAzwBMAOYAVQD1AKUAMQDcAAcARgDHAEYAvADuAEYAVgCGANcAVQDsAA4ARwBGAIwApgDHABoAUwCGAN4ATwB3AAwASwBVAK4AEgBdAHkAQABRAJkAqADiAJ4AVABGAKkATADxAEkAkgC0ABMA5wCbAEoAVgDzAA8AkgB/AIcA6QCuAE4AIwA5AJgAzACjAC8AYwBYANgAPQBcAL0AUwBnAFcA0wDZAEwAWQCPAM4AaQBkABoAPwBvAAYAKwBrAHUAPgBqAH4AUABuAF8AbgB5AGwAbQB8AG4AfQBcAF0AYQBnAFsAcQBXAF0AbABiAHoAagBtAGIAaQBuAG0AZgBsAHkAfQB3AF8AYgBmAHUAcQByAG8AagBlAHYAZABrAHcAeAB2AGIAcABrAHIAawByAHYAZwByAHUAZwBkAG4AawAeAHIAfQBqAGIAag"


    let images = [] 

    // get max number of array
    const N = Math.max(...props.n_samples)

    for(let i=0; i<props.category_names.length;i++)
        {let stack = []
        for(let n=0; n<N;n++){
            if(n<props.n_samples[i])
                stack.push(<CardMedia key={n} component='img' src={props.images[i][n]} />)
            else        
                stack.push(<CardMedia key={n} component='img' src={blank_image}/>)
        }

        images.push(stack)}
    return (            
        <Grid>
                {images.map((stack, index) => (
                   <Grid item xs={12} key={index}>
                   <Stack direction="column" spacing={2}>
                     <Typography variant="h6" align="left" style={{ marginTop: '16px' }}>
                       {props.category_names[index]}
                     </Typography>
                     <Stack direction="row" spacing={2} >
                       {stack}
                     </Stack>
                   </Stack>
                 </Grid>            
            ))}
        </Grid>
    )
}