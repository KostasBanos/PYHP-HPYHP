# -*- coding: utf-8 -*-
"""
Sequential Monte Carlo (SMC) Sampling for Pitman-Yor Hawkes Process
====================================================================

This module implements a Sequential Monte Carlo algorithm for the Pitman-Yor Hawkes Process,
which is used for modeling temporal clustering of events with textual content.
The algorithm combines temporal dynamics (Hawkes process) with textual similarity
using a discount parameter for the Pitman-Yor process.

Key Features:
- Sequential Monte Carlo particle filtering
- Pitman-Yor Hawkes Process with discount parameter
- Temporal triggering kernels with RBF basis functions
- Textual similarity modeling using Dirichlet-multinomial distributions
- Particle resampling to maintain diversity

Author: [Konstantinos Banos]
Date: [2025]
"""

from __future__ import print_function
from __future__ import division
import pickle
import numpy as np
from utils import *
import concurrent.futures
from functools import partial
import copyreg
import types
from copy import deepcopy
import json
import gc

def _pickle_method(m):
	"""
	Helper function to enable pickling of bound methods.
	This is required for multiprocessing support when using ProcessPoolExecutor.
	
	Args:
		m: A bound method to be pickled
		
	Returns:
		tuple: (getattr, (instance_or_class, method_name)) for reconstruction
	"""
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

# Register the pickle method for multiprocessing support
copyreg.pickle(types.MethodType, _pickle_method)

class Pitman_Yor_Hawkes_Process(object):
	"""
	Pitman-Yor Hawkes Process with Sequential Monte Carlo Sampling
	
	This class implements a Pitman-Yor Hawkes Process using Sequential Monte Carlo
	(SMC) methods for online inference. The process models temporal clustering of
	events with textual content, combining:
	- Temporal dynamics via Hawkes process triggering kernels
	- Textual similarity via Dirichlet-multinomial distributions
	- Pitman-Yor discount parameter for cluster creation
	
	The algorithm maintains a set of particles, each representing a possible
	state of the clustering process, and updates them sequentially as new
	documents arrive.
	"""
	
	def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth, sample_num, discount_param):
		"""
		Initialize the Pitman-Yor Hawkes Process with SMC sampling.
		
		Args:
			particle_num (int): Number of particles for SMC sampling
			base_intensity (float): Base intensity λ₀ for new cluster creation
			theta0 (np.array): Prior parameters for Dirichlet distribution over vocabulary
			alpha0 (np.array): Prior parameters for triggering kernel weights
			reference_time (np.array): Reference times for RBF kernel basis functions
			vocabulary_size (int): Size of the vocabulary
			bandwidth (np.array): Bandwidth parameters for RBF kernels
			sample_num (int): Number of samples for parameter estimation
			discount_param (float): Discount parameter d for Pitman-Yor process
		"""
		super(Pitman_Yor_Hawkes_Process, self).__init__()
		
		# Store all parameters
		self.particle_num = particle_num
		self.base_intensity = base_intensity
		self.theta0 = theta0
		self.alpha0 = alpha0
		self.reference_time = reference_time
		self.vocabulary_size = vocabulary_size
		self.bandwidth = bandwidth
		self.sample_num = sample_num
		self.discount_param = discount_param
		
		# Initialize particles with equal weights
		self.particles = []
		for i in range(particle_num):
			self.particles.append(Particle(weight = 1.0 / self.particle_num))
		
		# Pre-sample alpha parameters and their log priors for efficient computation
		alphas = []; log_priors = []
		for _ in range(sample_num):
			alpha = dirichlet(alpha0)  # Sample from Dirichlet prior
			log_prior = log_Dirichlet_CDF(alpha, alpha0)  # Compute log prior
			alphas.append(alpha); log_priors.append(log_prior)
		self.alphas = np.array(alphas)
		self.log_priors = np.array(log_priors)
		
		# Active time interval for efficient computation [tu, tn]
		self.active_interval = None

	def sequential_monte_carlo(self, doc, threshold):
		"""
		Main SMC algorithm: process a new document and update all particles.
		
		This method implements the core Sequential Monte Carlo algorithm:
		1. Compute active time interval for efficient computation
		2. Update each particle by sampling cluster assignment
		3. Resample particles based on their weights
		4. Perform garbage collection periodically
		
		Args:
			doc (Document): The new document to process
			threshold (float): Resampling threshold for particle weights
		"""
		print('\n\nhandling document %d' %doc.index)
		
		if isinstance(doc, Document):  # Process document with exact timing
			# Compute active time interval [tu, tn] for efficient computation
			# tu is the earliest time we need to consider based on kernel decay
			tu = EfficientImplementation(doc.timestamp, self.reference_time, self.bandwidth)
			self.active_interval = [tu, doc.timestamp]
			print('active_interval', self.active_interval)
			
			# Sequential particle update (currently used)
			# Update each particle by sampling cluster assignment and updating weights
			particles = []
			for particle in self.particles:
				particles.append(self.particle_sampler(particle, doc))
			self.particles = particles

			# Alternative parallel implementation with ProcessPoolExecutor 
			# This would use multiprocessing for faster computation but requires
			# careful handling of shared state and pickling
			'''
			partial_particle_sampler = partial(self.particle_sampler, doc = doc)
			with concurrent.futures.ProcessPoolExecutor(max_workers = self.particle_num) as executor:
				self.particles = list(executor.map(partial_particle_sampler, self.particles))
			'''

			# Another parallel implementation approach 
			'''
			particle_index = 0
			partial_particle_sampler = partial(self.particle_sampler, doc = doc)
			executor = concurrent.futures.ProcessPoolExecutor(max_workers = self.particle_num)
			wait_for = [executor.submit(partial_particle_sampler, particle) for particle in self.particles]
			concurrent.futures.wait(wait_for)
			particles = []
			for f in concurrent.futures.as_completed(wait_for):
				particle = f.result()
				particles.append(particle)
			self.particles = particles
			'''
			
			# Perform particle resampling based on weights
			self.particles = self.particles_normal_resampling(self.particles, threshold)
			
			# Periodic garbage collection to manage memory usage
			if (doc.index+1) % 100 == 0:
				gc.collect()
		else:  # Handle other document types (currently not implemented)
			print('deal with the case of exact timing')

	def particle_sampler(self, particle, doc):
		"""
		Update a single particle by processing a new document.
		
		This method performs three key operations for each particle:
		1. Sample cluster assignment for the new document
		2. Update the triggering kernel parameters for the selected cluster
		3. Calculate the log update probability for weight computation
		
		Args:
			particle (Particle): The particle to update
			doc (Document): The new document to process
			
		Returns:
			Particle: Updated particle with new cluster assignment and weights
		"""
		# Step 1: Sample cluster assignment for the new document
		particle, selected_cluster_index = self.sampling_cluster_label(particle, doc)
		
		# Step 2: Update the triggering kernel parameters for the selected cluster
		particle.clusters[selected_cluster_index].alpha = self.parameter_estimation(particle, selected_cluster_index)
		
		# Step 3: Calculate the log update probability for particle weight computation
		particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index, doc)
		
		return particle

	def sampling_cluster_label(self, particle, doc):
		"""
		Sample cluster assignment for a new document using Pitman-Yor Hawkes Process.
		
		This method implements the core cluster assignment logic that combines:
		- Temporal dynamics via Hawkes process triggering kernels
		- Textual similarity via Dirichlet-multinomial distributions  
		- Pitman-Yor discount parameter for cluster creation
		
		The assignment probability is proportional to:
		P(cluster k | doc) ∝ w_k(t) × P_textual(doc | cluster k)
		
		where w_k(t) are the discounted temporal weights and P_textual is the
		textual similarity probability.
		
		Args:
			particle (Particle): The particle containing current cluster state
			doc (Document): The new document to assign to a cluster
			
		Returns:
			tuple: (updated_particle, selected_cluster_index)
		"""
		if len(particle.clusters) == 0:  # First document case
			# Create the first cluster
			particle.cluster_num_by_now += 1
			selected_cluster_index = particle.cluster_num_by_now
			selected_cluster = Cluster(index = selected_cluster_index)
			selected_cluster.add_document(doc)
			particle.clusters[selected_cluster_index] = selected_cluster
			particle.docs2cluster_ID.append(selected_cluster_index)
			# Update active clusters for temporal tracking
			particle.active_clusters = self.update_active_clusters(particle)

		else:  # Subsequent documents case
			# Initialize cluster options: 0 for new cluster, existing clusters
			active_cluster_indexes = [0]  # 0 represents creating a new cluster
			active_cluster_rates = [self.base_intensity]  # Base intensity for new cluster
			
			# Compute textual probability for new cluster (self-similarity)
			cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(
				doc.word_distribution, doc.word_distribution,
				doc.word_count, doc.word_count, self.vocabulary_size, self.theta0)
			active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
			
			# Update active clusters based on temporal decay
			particle.active_clusters = self.update_active_clusters(particle)
			
			# Compute temporal rates and textual probabilities for existing clusters
			for active_cluster_index, timeseq in particle.active_clusters.items():
				active_cluster_indexes.append(active_cluster_index)
				
				# Compute temporal triggering rate using Hawkes process
				time_intervals = doc.timestamp - np.array(timeseq)
				alpha = particle.clusters[active_cluster_index].alpha
				rate = triggering_kernel(alpha, self.reference_time, time_intervals, self.bandwidth)
				active_cluster_rates.append(rate)
				
				# Compute textual similarity probability
				cls_word_distribution = particle.clusters[active_cluster_index].word_distribution + doc.word_distribution
				cls_word_count = particle.clusters[active_cluster_index].word_count + doc.word_count
				cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(
					cls_word_distribution, doc.word_distribution,
					cls_word_count, doc.word_count, self.vocabulary_size, self.theta0)
				active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)
			
			# print('active_cluster_indexes', active_cluster_indexes)
			# print('active_cluster_rates', active_cluster_rates)
			# print('active_cluster_textual_probs', active_cluster_textual_probs)
			
			# Apply Pitman-Yor discount parameter to temporal rates
			# Separate base intensity from existing cluster intensities
			# Convert to numpy arrays for vectorized operations
			existing_intensities = np.array(active_cluster_rates[1:])  # [λθ₁, λθ₂, ...]
			
			# Vectorized Pitman-Yor discount operations
			# wₖ(t) = max(0, λθₖ(t) - d) for existing clusters
			discounted_existing_cluster_rates = np.maximum(0, existing_intensities - self.discount_param)
			
			# w₀(t) = λ₀ + Σⱼ min(d, λθⱼ(t)) for new cluster
			min_discount_sum = np.sum(np.minimum(self.discount_param, existing_intensities))
			new_discounted_cluster_rate = self.base_intensity + min_discount_sum
			
			# Create final discounted rates (vectorized)
			active_discounted_cluster_rates = np.append([new_discounted_cluster_rate], discounted_existing_cluster_rates)
			
		
			print('active_discounted_cluster_rates', active_discounted_cluster_rates)
			
			# Filter out inactive clusters (zero rates after discount)
			# Keep only clusters with positive rates
			active_mask = np.array(active_discounted_cluster_rates) > 0
			active_cluster_indexes_filtered = np.array(active_cluster_indexes)[active_mask]
			active_discounted_cluster_rates_filtered = np.array(active_discounted_cluster_rates)[active_mask]
			active_cluster_textual_probs_filtered = np.array(active_cluster_textual_probs)[active_mask]
			
			print('active_cluster_indexes_filtered', active_cluster_indexes_filtered)
			print('active_discounted_cluster_rates_filtered', active_discounted_cluster_rates_filtered)
	
			# Combine temporal and textual probabilities in log space
			active_cluster_logrates = np.log(active_discounted_cluster_rates_filtered)
			cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs_filtered  # Log space combination
			cluster_selection_probs = cluster_selection_probs - np.max(cluster_selection_probs)  # Prevent overflow
			cluster_selection_probs = np.exp(cluster_selection_probs)  # Convert to probability space
			cluster_selection_probs = cluster_selection_probs / np.sum(cluster_selection_probs)  # Normalize
			print('cluster_selection_probs', cluster_selection_probs)

			# Sample cluster assignment using multinomial distribution
			np.random.seed()
			selected_cluster_array = multinomial(exp_num = 1, probabilities = cluster_selection_probs)
			selected_cluster_index = np.array(active_cluster_indexes_filtered)[np.nonzero(selected_cluster_array)][0]
			print('selected_cluster_index', selected_cluster_index)
			
			if selected_cluster_index == 0:  # Create new cluster
				particle.cluster_num_by_now += 1
				selected_cluster_index = particle.cluster_num_by_now
				selected_cluster = Cluster(index = selected_cluster_index)
				selected_cluster.add_document(doc)
				particle.clusters[selected_cluster_index] = selected_cluster
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index] = [self.active_interval[1]]  # Add current time
			else:  # Assign to existing cluster
				selected_cluster = particle.clusters[selected_cluster_index]
				selected_cluster.add_document(doc)
				particle.docs2cluster_ID.append(selected_cluster_index)
				#print('active_clusters', particle.active_clusters); print('cluster_num_by_now', particle.cluster_num_by_now) # FOR DEBUG
				particle.active_clusters[selected_cluster_index].append(self.active_interval[1])  # Add current time
		
		return particle, selected_cluster_index

	def parameter_estimation(self, particle, selected_cluster_index):
		"""
		Estimate triggering kernel parameters for a cluster using Bayesian inference.
		
		This method updates the alpha parameters (weights for RBF basis functions)
		of the triggering kernel for the selected cluster. It uses Sequential Monte Carlo
		to approximate the posterior distribution over alpha parameters given the
		temporal sequence of events in the cluster.
		
		For a new cluster (single event), it samples from the prior.
		For existing clusters, it uses importance sampling with pre-computed
		alpha samples and their likelihoods.
		
		Args:
			particle (Particle): The particle containing cluster information
			selected_cluster_index (int): Index of the cluster to update
			
		Returns:
			np.array: Updated alpha parameters for the triggering kernel
		"""
		# Get the temporal sequence of events in this cluster
		timeseq = np.array(particle.active_clusters[selected_cluster_index])
		
		if len(timeseq) == 1:  # First document in a new cluster
			# Sample alpha parameters from the prior distribution
			np.random.seed()
			alpha = dirichlet(self.alpha0)
			return alpha
		
		
		T = self.active_interval[1] + 1  # Current time + 1 for numerical stability
		alpha = update_triggering_kernel(timeseq, self.alphas, self.reference_time, 
										self.bandwidth, self.base_intensity, T, self.log_priors)
		return alpha

	def update_active_clusters(self, particle):
		"""
		Update active clusters by removing temporally inactive ones.
		
		This method implements efficient computation by maintaining only clusters
		that have events within the active time interval [tu, tn]. Clusters with
		all events before tu are removed since their contribution to the current
		event's intensity is negligible due to temporal decay.
		
		Args:
			particle (Particle): The particle whose active clusters to update
			
		Returns:
			dict: Updated active_clusters dictionary with only temporally relevant clusters
		"""
		if not particle.active_clusters:  # First document case
			# Initialize with the first cluster at current time
			particle.active_clusters[1] = [self.active_interval[1]]
		else:  # Update existing active clusters
			tu = self.active_interval[0]  # Earliest time to consider
			
			# Remove clusters that have no events after tu
			for cluster_index in list(particle.active_clusters.keys()):
				timeseq = particle.active_clusters[cluster_index]
				# Keep only events that occurred after tu
				active_timeseq = [t for t in timeseq if t > tu]
				
				if not active_timeseq:  # No active events
					del particle.active_clusters[cluster_index]
				else:  # Update with active events only
					particle.active_clusters[cluster_index] = active_timeseq
		
		return particle.active_clusters
	
	def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
		"""
		Calculate the log update probability for particle weight computation.
		
		This method computes the log probability of observing the new document
		given the selected cluster's current word distribution. This probability
		is used to update the particle's weight in the Sequential Monte Carlo
		algorithm.
		
		The probability is computed using the Dirichlet-multinomial distribution,
		which models the textual similarity between the document and the cluster.
		
		Args:
			particle (Particle): The particle containing cluster information
			selected_cluster_index (int): Index of the selected cluster
			doc (Document): The new document
			
		Returns:
			float: Log probability of the document given the cluster
		"""
		# Get cluster and document word distributions
		cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
		cls_word_count = particle.clusters[selected_cluster_index].word_count
		doc_word_distribution = doc.word_distribution
		doc_word_count = doc.word_count
		
		# Verify consistency of word counts
		assert doc_word_count == np.sum(doc.word_distribution)
		assert cls_word_count == np.sum(particle.clusters[selected_cluster_index].word_distribution)
		
		# Compute log probability using Dirichlet-multinomial distribution
		log_update_prob = log_dirichlet_multinomial_distribution(
			cls_word_distribution, doc_word_distribution,
			cls_word_count, doc_word_count, self.vocabulary_size, self.theta0)
		
		print('log_update_prob', log_update_prob)
		return log_update_prob

	def particles_normal_resampling(self, particles, threshold):
		"""
		Perform systematic resampling of particles based on their weights.
		
		This method implements the resampling step of Sequential Monte Carlo.
		Particles with low weights (below threshold) are removed and replaced
		by copies of particles with high weights. This prevents particle
		degeneracy and maintains diversity in the particle set.
		
		The resampling process:
		1. Update particle weights using log update probabilities
		2. Identify particles below the resampling threshold
		3. Resample low-weight particles from high-weight particles
		4. Normalize weights to maintain probability distribution
		
		Args:
			particles (list): List of particles to resample
			threshold (float): Weight threshold below which particles are resampled
			
		Returns:
			list: Resampled particles with updated weights
		"""
		print('\nparticles_normal_resampling')
		
		# Extract current weights and log update probabilities
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)
		
		weights = np.array(weights); log_update_probs = np.array(log_update_probs)
		print('weights before update:', weights)
		print('log_update_probs', log_update_probs)
		
		# Update weights using log probabilities (prevent overflow)
		log_update_probs = log_update_probs - np.max(log_update_probs)  # Numerical stability
		update_probs = np.exp(log_update_probs)  # Convert to probability space
		weights = weights * update_probs  # Update weights
		weights = weights / np.sum(weights)  # Normalize
		
		# Count particles that need resampling
		resample_num = len(np.where(weights + 1e-5 < threshold)[0])
		print('weights:', weights)
		print('resample_num:', resample_num)
		
		if resample_num == 0:  # No resampling needed
			# Just update particle weights
			for i, particle in enumerate(particles):
				particle.weight = weights[i]
			return particles
		else:  # Perform resampling
			# Identify particles to keep (above threshold)
			remaining_particles = [particle for i, particle in enumerate(particles) 
								  if weights[i] + 1e-5 > threshold]
			resample_probs = weights[np.where(weights > threshold + 1e-5)]
			resample_probs = resample_probs / np.sum(resample_probs)  # Normalize
			remaining_particle_weights = weights[np.where(weights > threshold + 1e-5)]
			
			# Update weights of remaining particles
			for i, _ in enumerate(remaining_particles):
				remaining_particles[i].weight = remaining_particle_weights[i]
			
			# Sample how many copies of each remaining particle to create
			np.random.seed()
			resample_distribution = multinomial(exp_num = resample_num, probabilities = resample_probs)
			
			if not resample_distribution.shape:  # Only one particle left
				# Create multiple copies of the single remaining particle
				for _ in range(resample_num):
					new_particle = deepcopy(remaining_particles[0])
					remaining_particles.append(new_particle)
			else:  # Multiple particles left
				# Create copies based on resample distribution
				for i, resample_times in enumerate(resample_distribution):
					for _ in range(resample_times):
						new_particle = deepcopy(remaining_particles[i])
						remaining_particles.append(new_particle)
			
			# Normalize weights of all particles
			update_weights = np.array([particle.weight for particle in remaining_particles])
			update_weights = update_weights / np.sum(update_weights)
			
			for i, particle in enumerate(remaining_particles):
				particle.weight = update_weights[i]
			
			# Verify constraints
			assert np.abs(np.sum(update_weights) - 1) < 1e-5  # Weights sum to 1
			assert len(remaining_particles) == self.particle_num  # Correct number of particles
			
			# Clean up memory
			self.particles = None
			return remaining_particles


def parse_newsitem_2_doc(news_item, vocabulary_size):
	"""
	Convert a news item tuple to a Document object.
	
	This utility function transforms raw news data into the Document format
	required by the Sequential Monte Carlo algorithm. It handles:
	- Index assignment for document identification
	- Timestamp conversion from Unix time to hours
	- Word distribution creation from word ID and count
	- Word count validation
	
	Args:
		news_item (tuple): Raw news item in format (id, timestamp, (word_id, count), word_count)
		vocabulary_size (int): Size of the vocabulary for word distribution vector
		
	Returns:
		Document: Document object ready for SMC processing
	"""
	# Extract components from news item tuple
	index = news_item[0]  # Document ID
	timestamp = news_item[1] / 3600.0  # Convert Unix timestamp to hours
	word_id = news_item[2][0]  # Word ID
	count = news_item[2][1]  # Word count
	
	# Create word distribution vector (sparse representation)
	word_distribution = np.zeros(vocabulary_size)
	word_distribution[word_id] = count
	
	word_count = news_item[3]  # Total word count
	
	# Create and return Document object
	doc = Document(index, timestamp, word_distribution, word_count)
	return doc

def main():
	"""
	Main function to run the Sequential Monte Carlo algorithm on news data.
	
	This function:
	1. Loads news data from JSON file
	2. Initializes the Pitman-Yor Hawkes Process with specified parameters
	3. Processes each news item sequentially using SMC
	4. Saves the final particle states for analysis
	
	The algorithm processes news articles to discover temporal clusters
	based on both temporal proximity and textual similarity.
	"""
	# Load news data from JSON file
	with open('./data/all_the_news_2017.json') as f:
		news_items = json.load(f)
	print('finish extracting news from json...')
	
	# Algorithm parameters
	vocabulary_size = 56720  # Size of vocabulary
	particle_num = 8  # Number of particles for SMC
	base_intensity = 0.1  # Base intensity λ₀ for new cluster creation
	discount_param = 0.05  # Pitman-Yor discount parameter d
	
	# Prior parameters
	theta0 = np.array([0.01] * vocabulary_size)  # Dirichlet prior for vocabulary
	alpha0 = np.array([0.1] * 4)  # Dirichlet prior for triggering kernel weights
	
	# Temporal kernel parameters
	reference_time = np.array([3, 7, 11, 24])  # Reference times (hours)
	bandwidth = np.array([5, 5, 5, 5])  # Bandwidth parameters for RBF kernels
	
	# Sampling parameters
	sample_num = 2000  # Number of samples for parameter estimation
	threshold = 1.0 / particle_num  # Resampling threshold
	
	# Initialize the Pitman-Yor Hawkes Process
	PYHP = Pitman_Yor_Hawkes_Process(
		particle_num = particle_num, 
		base_intensity = base_intensity, 
		theta0 = theta0, 
		alpha0 = alpha0,
		reference_time = reference_time, 
		vocabulary_size = vocabulary_size, 
		bandwidth = bandwidth, 
		sample_num = sample_num, 
		discount_param = discount_param)

	# Process each news item sequentially
	print("Starting Sequential Monte Carlo processing...")
	for news_item in news_items:
		# Convert news item to Document format
		doc = parse_newsitem_2_doc(news_item = news_item, vocabulary_size = vocabulary_size)
		# Process document using SMC algorithm
		PYHP.sequential_monte_carlo(doc, threshold)

	# Save final particle states for analysis
	print("Saving results...")
	with open('./results/particles.pkl', 'wb') as w:
		pickle.dump(PYHP.particles, w)
	
	print("Sequential Monte Carlo processing completed!")
	print(f"Processed {len(news_items)} documents")
	print(f"Final particles saved to ./results/particles.pkl")


# Execute main function when script is run directly
if __name__ == '__main__':
	main()