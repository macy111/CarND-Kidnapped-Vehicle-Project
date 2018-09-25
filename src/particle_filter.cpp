/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <float.h>

#include "particle_filter.h"
#define MIN_VALUE 1e-8

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for(int i = 0; i<num_particles; i++){
		double x_ran = dist_x(gen);
		double y_ran = dist_y(gen);
		double theta_ran = dist_theta(gen);
                Particle p{
			.id = i,
			.x = x_ran,
			.y = y_ran,
			.theta = theta_ran,
			.weight = 1.0
		};
		particles.push_back(p);
                weights.push_back(1.0);	
		//cout<<"create p "<<i<<" x:"<<x_ran<<" y:"<<y_ran <<endl;	
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	for(int i = 0; i<num_particles; i++){
		if(abs(yaw_rate) > MIN_VALUE){
			particles[i].x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
		}else{
			particles[i].x = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
			particles[i].y = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
		}
		
		
		particles[i].theta = particles[i].theta + yaw_rate*delta_t;
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
		//cout<<"predict p "<<i<<" x:"<<particles[i].x<<" y:"<<particles[i].y <<" theta:"<< particles[i].theta<< " yaw_rate:"<<yaw_rate<<" velocity:"<<velocity <<endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i<num_particles; i++){
		double w_all = 1.0;
		vector<int> associations_i;
		vector<double> sense_x_i;
		vector<double> sense_y_i;
		for(unsigned int j = 0; j < observations.size(); j++){
			double x_particle = particles[i].x;
			double y_particle = particles[i].y;
			double theta_particle = particles[i].theta;
			double x_obs = observations[j].x;
			double y_obs = observations[j].y;
    			double x_map = x_obs * cos(theta_particle) - y_obs * sin(theta_particle) + x_particle;
    			double y_map = y_obs * cos(theta_particle) + x_obs * sin(theta_particle) + y_particle; 
			//cout<<"x_map:"<<x_map<<" y_map:"<<y_map<<" x_obs:"<<x_obs<<" y_obs:"<<y_obs<<" theta_particle:"<<theta_particle<<" x_particle:"<<x_particle<<" y_particle:"<<y_particle<<endl;
			double nearest_dist = DBL_MAX;
			int nearest_id;
			double nearest_x = DBL_MAX;
			double nearest_y = DBL_MAX;
			for(unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++){
				double x_k_lm = map_landmarks.landmark_list[k].x_f;
				double y_k_lm = map_landmarks.landmark_list[k].y_f;
				double distance = dist(x_map, y_map, x_k_lm, y_k_lm);
				if (distance < nearest_dist){
					nearest_dist = distance;
					nearest_id = map_landmarks.landmark_list[k].id_i;
					nearest_x = x_k_lm;
					nearest_y = y_k_lm;
				}
				
			}
			//caculate prob
			double gauss_norm= (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));
			double exponent= (pow((x_map - nearest_x),2))/(2 * std_landmark[0] * std_landmark[0]) + (pow((y_map - nearest_y),2))/(2 * std_landmark[1]*std_landmark[1]);
			double weight= gauss_norm * exp(-exponent);
			w_all *= weight;
			associations_i.push_back(nearest_id);
			sense_x_i.push_back(x_map);
			sense_y_i.push_back(y_map);
			//cout<< "updateWeights p:"<< i << " ob:"<<j <<" n_id:" <<nearest_id<<" x_map:"<<x_map<<" y_map:"<<y_map <<" weight:"<<weight<<" w_all:"<<w_all<<endl;

		}
		particles[i].weight = w_all;
		weights[i] = w_all;
		particles[i].associations = associations_i;
		particles[i].sense_x = sense_x_i;
		particles[i].sense_y = sense_y_i;
		//cout<< "association p:"<< i << " "<<getAssociations(particles[i]) <<endl;
		//cout<< "getSenseX p:"<< i << " " << getSenseX(particles[i]) <<endl;
		//cout<< "getSenseY p:"<< i << " " << getSenseY(particles[i]) <<endl;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<> d(begin(weights), end(weights));
	vector<Particle> resample_p;
	for(int i = 0; i<num_particles; i++){
		int gen_i = d(gen);
		resample_p.push_back(particles[gen_i]);
		resample_p[i].id = i;
		//cout<<"resample_p p "<<i<<" x:"<<resample_p[i].x<<" y:"<<resample_p[i].y <<" theta:"<< resample_p[i].theta <<endl;
	}
	particles = resample_p;
	

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
