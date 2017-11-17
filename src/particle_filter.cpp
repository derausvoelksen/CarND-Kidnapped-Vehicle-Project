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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;

    default_random_engine gen;
    double std_x = std[0], std_y = std[1], std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; i++) {

        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        p.id = i;

        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std[], double v, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    double std_x = std[0], std_y = std[1], std_theta = std[2];
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    double rate = (v / yaw_rate);
    double yaw_change = yaw_rate * delta_t;
    
    for (auto &p : particles) {
        //yaw rate is zero
        if (fabs(yaw_rate) < 1e-5) {
            p.x += v * cos(p.theta) * delta_t;
            p.y += v * sin(p.theta) * delta_t;
        } else {
            p.x += rate * (sin(p.theta + yaw_change) - sin(p.theta));
            p.y += rate * (-cos(p.theta + yaw_change) + cos(p.theta));
            p.theta += yaw_change;
        }
        
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for (auto &o : observations) {

        double min_distance = 1e+9;
        
        for (auto &p : predicted) {

            auto distance = sqrt(pow(p.x - o.x, 2) + pow(p.y - o.y, 2));
            
            if (distance < min_distance) {
                o.id = p.id;
                min_distance = distance;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

    double coeff = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    double std_coeff_x = 2 * std_landmark[0] * std_landmark[0];
    double std_coeff_y = 2 * std_landmark[1] * std_landmark[1];
    weights.clear();

    vector<LandmarkObs> predicted;
    
    for (auto &p : map_landmarks.landmark_list) {
        LandmarkObs l;
        l.id = p.id_i;
        l.x = p.x_f;
        l.y = p.y_f;
        predicted.push_back(l);
    }

    for (int i = 0; i < particles.size(); i++) {

        auto &particle = particles[i];

        vector<LandmarkObs> obs;
        
        for (auto &l : observations) {
            LandmarkObs transformed;
            transformed.y = particle.y + l.x * sin(particle.theta) + l.y * cos(particle.theta);
            transformed.x = particle.x + l.x * cos(particle.theta) - l.y * sin(particle.theta);
            obs.push_back(transformed);
        }

        dataAssociation(predicted, obs);

        particle.weight = 1;
        
        for (auto &o : obs) {

            LandmarkObs desired;

            for (auto &p : predicted) {
                if (o.id == p.id) {
                    desired = p;
                    break;
                }
            }
            particle.weight *= coeff * exp(-(pow(o.x - desired.x, 2) / std_coeff_x + pow(o.y - desired.y, 2) / std_coeff_y));
        }

        weights.push_back(particle.weight);
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    discrete_distribution<> dist(weights.begin(), weights.end());
    vector<Particle> new_particles;
    
    for (int i = 0; i < particles.size(); i++) {
        int index = dist(gen);
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;


}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
