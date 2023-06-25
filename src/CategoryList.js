import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import {Box, Grid, Typography, Stack, TextField} from '@mui/material';

import {height, im_height, category_colors, max_categories} from './constants'
import LinearProgress from '@mui/material/LinearProgress';
import React from 'react';


export default function CategoryList(props){
    
    let names = Array(props.category_names.length)
    let sample_counter

    let topmargin = 0
    for(let i=0; i<props.category_names.length;i++)
        {
        if(props.n_samples[i]===0)
            sample_counter = "sin imágenes aún"
        if(props.n_samples[i]===1)
            sample_counter = "1 imagen"
        if(props.n_samples[i]>1)
            sample_counter = props.n_samples[i] + " imágenes"
        
        topmargin = props.ypos[0] - im_height/3    
        if(i>0)
           topmargin = topmargin + i*im_height/3

        names[i] = <Stack  key={i}>
                        <TextField style={{top: topmargin}} sx={{ input: {"font-size": "150%", color: category_colors[i]}, 
                        '& .MuiInput-underline:after': {borderBottomColor: category_colors[i]}, 
                        '& label.Mui-focused': {color: category_colors[i]},
                        '& label': {"font-size": "150%"},
                        }} 
                        label={"Objeto "+ i + ": " + sample_counter} value={props.category_names[i]} 
                        onFocus={()=>{props.enableKeys(false)}}
                        onBlur={()=>{props.enableKeys(true)}}
                        onChange={(e)=>{props.get_category_names(i, e.target.value)}} variant="standard" 
                        InputProps={{endAdornment: <IconButton aria-label="delete"  
                                    onClick={()=>{props.handleRemoveCategory(i)}} disabled={names.length>2?false:true}> <DeleteIcon/></IconButton>}}/> 
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                                <LinearProgress style={{top: topmargin, height: 15}} sx={{backgroundColor: "whitesmoke", "& .MuiLinearProgress-bar": {backgroundColor: category_colors[i]}}}
                                variant="determinate" value={props.scores[i]*100} />
                            </Box>
                            <Typography
                            style={{
                                position: "relative",
                                color: category_colors[i],
                                top: topmargin,
                                left: "5%",
                                transform: "translateX(-50%)",
                            }}
                            >
                            {Math.round(props.scores[i]*100)}% 
                            </Typography>
                        </Box>
        
                    </Stack>
    }   
    let add_item=""
    if (names.length<max_categories)
        add_item = <TextField disabled key={99} label={"Nuevo objeto"} value={""} variant="standard" style={{top: topmargin + 20}}
        InputProps={{endAdornment: <IconButton onClick={props.handleAddCategory}><AddIcon/></IconButton>
    }}/> 

    //imágenes ejemplo de cada clase
    const category_display = props.images.map((v, k) => <image key={k} xlinkHref={props.images[k]}
                                                               x={im_height/10} y={Math.floor(props.ypos[k])-im_height/2}
                                                               viewBox={"0 0 1 1"}
                                                               width={im_height} height={im_height} preserveAspectRatio="xMidYMid slice"/>) 

    return (    
        <Grid container>
            <Grid item sm={8}>
                {names} 
                <Grid >
                {add_item}
                </Grid>
            </Grid>
            <Grid item sm={4}>
                <svg  width={11*im_height/10} height={height} className={"shadow"}>
                    {category_display}
                </svg>
            </Grid>
        </Grid>
    )
}