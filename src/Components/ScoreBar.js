import React from 'react'
import ProgressBar from 'react-bootstrap/ProgressBar'
import './css/ScoreBar.css'

function ScoreBar(props){
    // const style = {
    //     backgroundColor : `rgba(${(1 - props.value) * 255}, ${props.value * 140}, ${10})`,
    // }
    const variant = (score) => {
        if (score < 0.25) {
            return 'danger'
        }
        else if(score < 0.5){
            return 'warning'
        }
        else if(score < 0.75){
            return 'info'
        }
        else {
            return 'success'
        }
    }
    return(
        <ProgressBar 
        id="progress" 
        now={props.value * 100} 
        label={`${props.value * 100}%`}
        //style={style}
        variant={variant(props.value)}
        />
    )
}

export default ScoreBar