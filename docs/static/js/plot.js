    
    $(document).ready(function() {

        // Real Robot Bar Chart
        var trace1 = {
            x: ['Pick Place Cup', 'Stamp The Paper', 'Take the Mug', 
            'Put The Mug On The Coaster', 'Roll The Stamp'],
            y: [90, 80, 90, 100, 70],
            name: 'Canonical',
            type: 'bar',
            marker: {
            color: '#007FA1'
            }
        };
    
        var trace2 = {
            x: ['Pick Place Cup', 'Stamp The Paper', 'Take the Mug', 
            'Put The Mug On The Coaster', 'Roll The Stamp'],
            y: [80, 60, 70, 70, 60],
            name: 'Camera-Shift',
            type: 'bar',
            marker: {
            color: '#7BDCB5'
            }
        };
    
        var trace3 = {
            x: ['Pick Place Cup', 'Stamp The Paper', 'Take the Mug', 
            'Put The Mug On The Coaster', 'Roll The Stamp'],
            y: [93.0, 87.0, 77.0, 77.0, 67.0],
            name: 'BackGround-Change',
            type: 'bar',
            marker: {
            color: '#00D084'
            }
        };
    
        var trace4 = {
            x: ['Pick Place Cup', 'Stamp The Paper', 'Take the Mug', 
            'Put The Mug On The Coaster', 'Roll The Stamp'],
            y: [83.0, 78.0, 72.0, 83.0, 56.0],
            name: 'New-Object',
            type: 'bar',
            marker: {
            color: '#FF5A5F'
            }
        };
    
        var data = [trace1, trace2, trace3, trace4];
    
        var layout = {
            barmode: 'group',
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: -0.2,
            xanchor: 'center',
            yanchor: 'top',
            // title: ' Success rates (%) of GROOT in realrobot tasks.',        
            showarrow: false,
    
        };
    
        Plotly.newPlot('real-robot-results-div', data, layout);
});