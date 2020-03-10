import React from 'react'
import ProgressBar from 'react-bootstrap/ProgressBar'
import './css/ScoreBar.css'

function ScoreBar(props){
    return(
        <ProgressBar id="progress" now={props.value * 100} label={`${props.value * 100}%`}/>
    )
}

export default ScoreBar