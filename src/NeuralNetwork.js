import { xpos, height, unit_sep, max_categories, category_colors, width } from "./constants"
import React from 'react';


function Neuron(props){

    let style = "visible"
    if(props.is_hidden){
        style += " hidden"
        }

    let transition_end = ()=>{}
    let neuron_face = "neuron-base "
    let color = "white"
    if (props.is_on){
        if(props.layer <= 1){
            neuron_face += "neuron-on"+props.layer.toString()
            color = props.color
        }
        if(props.layer === 2 && props.output_active) {
            transition_end = props.onTransitionEnd
            neuron_face += "neuron-on"+props.layer.toString()
            color = props.color
        }
    }
    
    return(
        <g  onClick={()=>{if(props.layer === 2 & !props.is_hidden) props.onClick(props.k_pos)}} className={style} onTransitionEnd={transition_end} transform={"translate(" + (props.x-25).toString() + " " + (props.y-25).toString() + ") scale(0.12)"}
            >
			<circle fill={"whitesmoke"} cx="245" cy="240.808" r="170"/>
			<path d="M280,190.808c-44.112,0-80,35.888-80,80s35.888,80,80,80c21.36,0,41.456-8.32,56.568-23.432l-11.312-11.312
				c-12.096,12.088-28.16,18.744-45.256,18.744c-35.288,0-64-28.712-64-64c0-35.288,28.712-64,64-64c35.288,0,64,28.712,64,64
				c0,12.728-3.728,25.024-10.776,35.544l13.296,8.904c8.816-13.168,13.48-28.544,13.48-44.448
				C360,226.696,324.112,190.808,280,190.808z"/>
			<path d="M475.416,328.248L448,347.44v-36.632h-16v47.832l-35.056,24.536l-6.072-5.064l-3.04-9.12h-0.008
				c-2.184-6.56-0.968-14.008,3.264-19.912C412.624,319.056,424,283.696,424,246.808c0-22.824-4.312-45.04-12.832-66.048
				c-3.76-9.264-3.168-20.056,1.6-29.608l1.624-3.256l15.344-12.784l27.192,20.392l7.312,29.24l15.512-3.872l-6.064-24.248
				l24.24-6.056l-3.872-15.512l-28.304,7.08l-23.392-17.544l16.536-13.784H496v-16h-32V47.6l7.84-39.216L456.152,5.24l-6.224,31.12
				l-39.4-13.136l-5.064,15.168l42.664,14.224L448,99.048l-43.288,36.072c-11.36,1.256-22.688-2.656-30.648-10.864
				c-26.176-26.936-59.48-44.504-96.304-50.784c-12.76-2.176-23.728-9.92-29.768-20.792V27.088l20.44-13.624L259.56,0.152
				L240,13.192l-19.56-13.04l-8.872,13.312L232,27.088v25.824l-1.04,2.08C225.624,65.664,216,73.456,204.568,76.36
				c-25.144,6.392-49.016,18.592-69.048,35.288c-10.312,8.6-23.56,11.432-35.696,7.904l-11.528-9.88l6.984-41.904l36.832-22.104
				l-8.232-13.72L92.432,50.816L79.584,12.272l-15.168,5.064l15.368,46.112l-5.68,34.056L48,75.128v-52.32H32v48H0v16h37.04
				l6.896,5.912l-32.048,19.232l8.232,13.72l36.624-21.976l32.368,27.744l3.384,10.136c2.936,8.824,2.112,18.648-2.344,27.68
				c-8.496,17.272-14.056,35.728-16.528,54.856c-0.424,3.296-2.624,6.248-5.896,7.88l-12.352,6.176l-18.48-6.16l-22.496-30
				l-12.8,9.6l19.84,26.448L2.344,257.152l11.312,11.312L34.16,247.96l5.84,1.944v49.592l-21.656,21.656l11.312,11.312l17.48-17.48
				L65.6,339.608l12.8-9.592L56,300.144v-44.392l11.736,5.872c3.264,1.632,5.472,4.592,5.896,7.928
				c6.304,48.76,33.12,92.976,73.568,121.32c3.008,2.104,4.8,5.672,4.8,9.544v12.176l-20.952,34.936l-40.984,6.832L60.8,432.408
				l-9.6,12.8l23.464,17.6l-23.464,17.6l9.6,12.8l30.44-22.832l39.728-6.624l13.608,34.024l14.856-5.936l-14.528-36.32
				l21.152-35.248l9.288-6.96c3.272-2.456,7.432-3.08,11.096-1.696c19.776,7.424,40.496,11.192,61.56,11.192
				c36.88,0,72.224-11.344,102.208-32.824c5.4-3.856,12.128-5.048,18.52-3.216l11.44,3.264l19.56,16.296l-13.8,41.376l-43.248,7.208
				l2.624,15.784l40.72-6.784l6.216,24.832l15.512-3.872l-7.432-29.744l12.6-37.8l11.784,9.816l7.448,37.232l15.688-3.144
				l-6.384-31.896l39.848-6.64l-2.624-15.784l-44.384,7.4l-24.616-20.52l74.904-52.432L475.416,328.248z M378.096,339.752
				c-6.592,9.192-8.856,20.664-6.304,31.272c-10.68-2.648-21.872-0.496-30.888,5.952c-27.264,19.52-59.384,29.832-92.904,29.832
				c-19.144,0-37.96-3.424-55.944-10.176c-3.128-1.168-6.376-1.752-9.608-1.752c-5.04,0-10.048,1.4-14.488,4.12
				c-0.424-8.544-4.68-16.408-11.568-21.232c-36.784-25.768-61.16-65.952-66.88-110.256c-1.12-8.632-6.584-16.176-14.616-20.192
				l-1.016-0.512l1-0.496c8.048-4.024,13.504-11.56,14.616-20.16c2.24-17.376,7.288-34.152,15.008-49.832
				c6.224-12.64,7.408-26.56,3.336-39.256c13.536,0.68,26.92-3.944,37.928-13.12c18.2-15.184,39.896-26.264,62.736-32.08
				c13-3.296,24.304-11.112,32.064-21.92c8.704,10.048,20.896,16.992,34.512,19.304c33.456,5.712,63.72,21.68,87.512,46.16
				c8.824,9.088,20.52,14.584,32.888,15.776c-3.928,11.768-3.68,24.384,0.864,35.592c7.736,19.08,11.656,39.28,11.656,60.032
				C408,280.328,397.656,312.472,378.096,339.752z"/>
			<rect x="224" y="110.808" width="16" height="16"/>
			<rect x="128" y="278.808" width="16" height="16"/>
			<rect x="248" y="374.808" width="16" height="16"/>
			<rect x="168" y="190.808" width="16" height="16"/>
			<rect x="152" y="134.808" width="16" height="16"/>
			<rect x="304" y="150.808" width="16" height="16"/>
			<circle className={neuron_face} fill={color} cx="280" cy="270.808" r="58"/>
		</g>

    );
}

// Network
export function Network(props){
    var nunits
    var layers = Array(3)
    var layers_face = Array(3)
    var ypos = Array(3)

    const max_units = [4, 5, max_categories]
    switch (props.size) {
        case 0:
            nunits = [3, 3, props.n_outputs]
            break
        case 1:
            nunits = [3, 4, props.n_outputs]
            break
        case 2:
            nunits = [3, 5, props.n_outputs]
            break
        default:
            break
    }
    
    for (let i = 0; i < layers.length; i++) {
        layers[i] = Array(max_units[i])
        layers_face[i] = Array(max_units[i])
        ypos[i] =  Array(max_units[i])
        for (let j = 0; j < max_units[i]; j++) {

            ypos[i][j] = height / 2 + unit_sep[i] * (j - nunits[i] / 2)
            const is_on = (props.category !== -1)
            let color = "white"
            if(is_on){color=category_colors[props.category]}
            layers[i][j] = <Neuron onClick={props.onClick} key={j} onTransitionEnd={props.onTransitionEnd} is_on={is_on}
                                   is_hidden={(j >= nunits[i])} layer={i} k_pos={j} output_active={props.category === j}
                                   x={xpos[i]} y={ypos[i][j]} color={color}/>


        }
    }
    // lines
    var lines = Array((max_units[0]+max_units[2]) * max_units[1])
    var ind = 0
    var style = 0
    let color = "black"
    for (var i = 0; i < layers.length-1; i++)
        for (var j = 0; j < max_units[i]; j++)
            for (var z = 0; z < max_units[i+1]; z++) {
                style = "line"
                color = "black"
                if (props.category !== -1) {
                    if (i === 0){ //first lines
                        color = category_colors[props.category]
                        style += " line-on1"
                    }
                    if (i === 1){ //second lines
                        if (z === props.category){
                            color = category_colors[props.category]
                            style += " line-on2"
                        }
                    }
                }
                if (j >= nunits[i] || z >= nunits[i+1])
                        style += " hidden"

                lines[ind] = <path key={ind} stroke={color}
                                   d={"M"+ xpos[i].toString() +" "+ ypos[i][j].toString()+" L" + xpos[i+1].toString() +" "+ ypos[i+1][z].toString()}
                                   className={style} strokeWidth={"1"} />
                ind++
            }
      
    
    return (
        <svg  width={width} height={height} className={"shadow"}>
            {lines}
            {layers}
        </svg>

    );
}
