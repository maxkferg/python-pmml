/* Load test script
 * Test each of the services to generate throughput vs response time
 * Written in node.js because this is a purely asuync type of test
 * Usage:
 * 		npm i
 *      node load.js
 */
var fs = require('fs');
var request = require('request');
var json2csv = require('json2csv');
var async = require('async');


// Google Cloud
var api = 'http://104.198.10.35/predict/'
var filename = 'results/simple-google.csv';
var throughput = [1,10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500];

// Local Machine
//var api = "http://localhost:5000/predict/";
//var filename = 'results/simple-macpro.csv';
//var throughput = [1,10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500]

// Raspberry PI
// var api = "http://10.34.189.71/predict/";
// var filename = 'results/simple-raspberry.csv';
// var throughput = [1,10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600]


var model = "simple-example.pmml"
var xnew = [1,4];

// Run the tests
async.mapSeries(throughput, runTest, function(err, results){
	var data = results.map(function(response,i){
		return {response:response,throughput:throughput[i]}
	});
	fs.writeFileSync(filename, json2csv({data:data}));
	console.log('Results saved to '+filename);
});


/* Run a single load test
 * @param(int) rate. The test throughput
 * @param(callback) callback. Function returning the average response time
 */
function runTest(rate,callback){
	var options = {
		url: api+model,
		method: 'post',
		json: {
			xnew: xnew
		}
	}
	var max = 10*rate;
	loadTest(options,rate,max,callback)
}



/* Load test a url.
 * @param(Object) options. The options to be passed to request
 * @param(int) rate. The throughput to test the function at
 * @param(int) max. The maximum number of requests
 */
function loadTest(options,rate,max,callback){
	var started  = 0;
	var finished = 0;
	var duration = 0;
	var interval = Math.round(1000/rate);
	var durations = [];

	var comp = setInterval(function(){
		var start = Date.now();
		// Start the actual request
		request(options, function(error, response, body) {
			console.log(body)
			duration = Date.now() - start;
			durations.push(duration);
			finished +=1;
			if (finished==max){
				finishTest(rate,durations);
				callback(undefined,mean(durations));
			}
		});
		// Have we started enough?
		started+=1;
		if (started==max){
			clearInterval(comp)
		}
	}, interval);
}




/* finishTest
 * End the load test
 * Prints test info
 */
function finishTest(rate,durations){
	console.log('Throughput: ' + rate + 'req/s, duration: '+mean(durations));
}


/* Return the mean of ann array of numbers
 * @param(array) vector. Some vector of numbers
 */
function mean(vector){
	var total = 0;
	for(var i = 0; i < vector.length; i++) {
	    total += vector[i];
	}
	return total / vector.length;
}




