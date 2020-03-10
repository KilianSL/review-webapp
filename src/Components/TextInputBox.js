import React, { Component } from 'react'
import './css/ScoreBar.css'

class TextInputBox extends Component{

    constructor(props){
        super(props)
        this.state = {
            value : props.value
        }
    }

    updateState(newval){
        this.setState({
            value : newval
        })
        this.props.callback(newval)
    }

    handleChange(event){
        this.updateState(event.target.value)
    }

    render(){
        return(
            <div className="form-group shadow-textarea">
                <textarea 
                className="form-control z-depth-1" 
                id="review" 
                rows="8" 
                placeholder="Write something here..." 
                onChange={event => this.handleChange(event)}
                />
            </div>
        )
    }
}

export default TextInputBox