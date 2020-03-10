import React, { Component } from 'react'
import TextInputBox from './TextInputBox'
import ScoreBar from './ScoreBar'
import './css/App.css'

class App extends Component{
    
    constructor(props){
        super(props)
        this.state = {
            score : 0
        }
    }

    submitScore = () => {
        this.setState({
            score : 0
        })
    }


    updateScore = (newvalue) => {
        console.log(newvalue)
        fetch('http://127.0.0.1:5000/?value=' + newvalue)
        .then(request => request.text())
        .then(newscore => this.setState({
            score : parseFloat(newscore).toFixed(2)
        }))
    }

    render(){
        return(
            <div className="form-review">
                <h1 className="h3 mb-3 font-weight-normal">What do you think of the website?</h1>
                <TextInputBox value = {this.state.value} callback={this.updateScore}/>
                <h5>Positivity Score</h5>
                <ScoreBar value = {this.state.score}/>
                <button className="btn btn-lg btn-primary btn-block" id="submit_button" onClick={this.submitScore}>
                    <span>Submit</span>
                    <i className="fa fa-paper-plane m-l-7"></i>
                </button>
            </div>
        )
    }
}

export default App