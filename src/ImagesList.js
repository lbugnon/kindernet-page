import {Grid, Stack, CardMedia} from '@mui/material';
import React from 'react';


export default function ImagesList(props){
        
    let images = [] 
    for(let i=0; i<props.category_names.length;i++)
        {let stack = []
        for(let n=0; n<props.n_samples[i];n++)
            stack.push(<CardMedia key={n} component='img' src={props.images[i][n]} />)      
        images.push(stack)}
    return (            
        <Grid container>
            <Stack spacing={2}>
                {images.map((stack, index) => (
                    <Stack direction="row" spacing={2} key={index}>
                    {stack}
                    </Stack>                
            ))}
            </Stack>

        </Grid>
    )
}