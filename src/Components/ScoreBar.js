import React from 'react'
import ProgressBar from 'react-bootstrap/ProgressBar'
import './css/ScoreBar.css'

function ScoreBar(props){
    const style = {
        backgroundColor : `rgba(${(1 - props.value) * 255}, ${props.value * 140}, ${10})`,
    }
    return(
        <ProgressBar 
        id="progress" 
        now={props.value * 100} 
        label={`${props.value * 100}%`}
        style={style}
        />
    )
}

export default ScoreBar