# -*- coding: utf-8 -*-
"""
Utility Functions for Sequential Monte Carlo Hawkes Process
===========================================================

This module contains essential utility functions and classes for implementing
the Sequential Monte Carlo algorithm for the Pitman-Yor Hawkes Process. It
includes data structures for documents, clusters, and particles, as well as
mathematical functions for probability distributions, temporal kernels, and
Bayesian inference.

Key Components:
- Data Structures: Document, Cluster, Particle classes
- Probability Distributions: Dirichlet, Multinomial sampling
- Temporal Kernels: RBF kernels for Hawkes process triggering
- Bayesian Inference: Parameter estimation and likelihood computation
- Textual Modeling: Dirichlet-multinomial distributions

Author: [Konstantinos Banos]
Date: [2025]
"""

from __future__ import division
import numpy as np
import scipy.stats
from scipy.special import erfc, gammaln
import pickle
from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional
from wordcloud import WordCloud


class Document(object):
	"""
	Document class representing a single document in the corpus.
	
	This class encapsulates all information about a document including its
	temporal information (timestamp) and textual content (word distribution).
	Documents are the basic units processed by the Sequential Monte Carlo
	algorithm for cluster assignment.
	
	Attributes:
		index (int): Unique identifier for the document
		timestamp (float): Time when the document was created (in hours)
		word_distribution (np.array): Vector of word counts across vocabulary
		word_count (int): Total number of words in the document
	"""
	
	def __init__(self, index, timestamp, word_distribution, word_count):
		"""
		Initialize a Document object.
		
		Args:
			index (int): Unique document identifier
			timestamp (float): Document creation time in hours
			word_distribution (np.array): Word count vector across vocabulary
			word_count (int): Total word count in the document
		"""
		super(Document, self).__init__()
		self.index = index
		self.timestamp = timestamp
		self.word_distribution = word_distribution
		self.word_count = word_count
		
class Cluster(object):
	"""
	Cluster class representing a group of related documents.
	
	This class maintains the state of a cluster including its temporal dynamics
	(alpha parameters for triggering kernel) and textual content (aggregated
	word distribution). Clusters are created and updated dynamically as
	documents are assigned to them.
	
	Attributes:
		index (int): Unique identifier for the cluster
		alpha (np.array): Parameters for the temporal triggering kernel
		word_distribution (np.array): Aggregated word counts across vocabulary
		word_count (int): Total word count across all documents in cluster
	"""
	
	def __init__(self, index):
		"""
		Initialize a Cluster object.
		
		Args:
			index (int): Unique cluster identifier
		"""
		super(Cluster, self).__init__()
		self.index = index
		self.alpha = None  # Temporal triggering kernel parameters
		self.word_distribution = None  # Aggregated word distribution
		self.word_count = 0  # Total word count

	def add_document(self, doc):
		"""
		Add a document to this cluster and update aggregated statistics.
		
		This method updates the cluster's word distribution and word count
		by incorporating the new document's textual content.
		
		Args:
			doc (Document): The document to add to the cluster
		"""
		if self.word_distribution is None:
			# First document in cluster - initialize word distribution
			self.word_distribution = np.copy(doc.word_distribution)
		else:
			# Add document's word counts to existing distribution
			self.word_distribution += doc.word_distribution
		
		# Update total word count
		self.word_count += doc.word_count

	def __repr__(self):
		"""
		Return string representation of the cluster.
		
		Returns:
			str: String representation showing cluster index and word count
		"""
		return 'cluster index:' + str(self.index) + '\n' + 'document index:' + str(self.documents) + '\n' +'word_count: ' + str(self.word_count) \
		+ '\nalpha:' + str(self.alpha)

class Particle(object):
	"""
	Particle class representing a single hypothesis in Sequential Monte Carlo.
	
	Each particle maintains a complete state of the clustering process including
	all clusters, document assignments, and temporal information. Particles are
	updated sequentially as new documents arrive and are resampled based on
	their likelihood weights.
	
	Attributes:
		weight (float): Importance weight of this particle
		log_update_prob (float): Log probability of the most recent update
		clusters (dict): Dictionary mapping cluster indices to Cluster objects
		docs2cluster_ID (list): Document-to-cluster assignments in order
		active_clusters (dict): Temporal information for efficient computation
		cluster_num_by_now (int): Total number of clusters created so far
	"""
	
	def __init__(self, weight):
		"""
		Initialize a Particle object.
		
		Args:
			weight (float): Initial importance weight of the particle
		"""
		super(Particle, self).__init__()
		self.weight = weight
		self.log_update_prob = 0  # Log probability of most recent update
		self.clusters = {}  # Dictionary: cluster_index -> Cluster object
		self.docs2cluster_ID = []  # List of cluster assignments for each document
		self.active_clusters = {}  # Dictionary: cluster_index -> list of timestamps
		self.cluster_num_by_now = 0  # Counter for total clusters created

	def __repr__(self):
		"""
		Return string representation of the particle.
		
		Returns:
			str: String representation showing document assignments and weight
		"""
		return 'particle document list to cluster IDs: ' + str(self.docs2cluster_ID) + '\n' + 'weight: ' + str(self.weight)
		

def dirichlet(prior):
	"""
	Draw samples from a Dirichlet distribution.
	
	This function samples from a Dirichlet distribution with given prior parameters
	and returns a probability vector that sums to 1. The Dirichlet distribution
	is commonly used as a prior for multinomial distributions in Bayesian inference.
	
	Args:
		prior (np.array): Concentration parameters for the Dirichlet distribution
		
	Returns:
		np.array: Probability vector sampled from Dirichlet distribution
	"""
	return np.random.dirichlet(prior).squeeze()

def multinomial(exp_num, probabilities):
	"""
	Draw samples from a multinomial distribution.
	
	This function samples from a multinomial distribution with specified number
	of experiments and probability vector. It returns a count vector indicating
	how many times each outcome occurred.
	
	Args:
		exp_num (int): Number of experiments/trials
		probabilities (np.array): Probability vector for each outcome
		
	Returns:
		np.array: Count vector indicating occurrences of each outcome
	"""
	result = np.random.multinomial(int(exp_num), probabilities)
	# Ensure we always return at least 1D array to avoid scalar issues
	if result.ndim == 0:
		return np.array([result])
	else:
		return result.squeeze()

def EfficientImplementation(tn, reference_time, bandwidth, epsilon = 1e-5):
	"""
	Compute the earliest time needed for efficient triggering kernel updates.
	
	This function implements an efficient computation strategy by determining
	the earliest time tu such that events before tu have negligible contribution
	to the current event's intensity. This allows pruning of old events to
	maintain computational efficiency.
	
	The computation is based on the exponential decay of RBF kernels and
	uses error tolerance epsilon to determine when contributions become negligible.
	
	Args:
		tn (float): Current document time
		reference_time (np.array): Reference times for RBF kernels
		bandwidth (np.array): Bandwidth parameters for RBF kernels
		epsilon (float): Error tolerance for pruning (default: 1e-5)
		
	Returns:
		float: Earliest time tu to consider for efficient computation
	"""
	max_ref_time = max(reference_time)
	max_bandwidth = max(bandwidth)
	
	# Compute tu using error tolerance and kernel decay properties
	tu = tn - (max_ref_time + np.sqrt(-2 * max_bandwidth * np.log(0.5 * epsilon * np.sqrt(2 * np.pi * max_bandwidth**2))))
	return tu

def log_Dirichlet_CDF(outcomes, prior):
	"""
	Compute log probability density of Dirichlet distribution (simplified case).
	
	This function computes the log probability density for a Dirichlet distribution
	in the special case where all prior parameters equal 1 (symmetric case).
	This simplification avoids complex integral computations while maintaining
	the essential properties needed for Bayesian inference.
	
	Args:
		outcomes (np.array): Observed probability vector
		prior (np.array): Prior parameters (must be all 1s for this implementation)
		
	Returns:
		float: Log probability density
	"""
	return np.sum(np.log(outcomes)) + scipy.stats.dirichlet.logpdf(outcomes, prior)

def RBF_kernel(reference_time, time_interval, bandwidth):
	"""
	Compute Radial Basis Function (RBF) kernel values for Hawkes process.
	
	This function implements Gaussian RBF kernels that model the temporal
	influence between events in a Hawkes process. The kernel captures how
	the influence of past events decays over time with Gaussian-like decay.
	
	The RBF kernel is defined as:
	K(t) = exp(-(t - μ)²/(2σ²)) / sqrt(2πσ²)
	
	where μ is the reference time and σ is the bandwidth.
	
	Args:
		reference_time (np.array): Reference times (μ) for the kernels
		time_interval (float/np.array): Time differences from reference times
		bandwidth (np.array): Bandwidth parameters (σ) for the kernels
		
	Returns:
		np.array: RBF kernel values for each reference time
	"""
	numerator = -(time_interval - reference_time) ** 2 / (2 * bandwidth ** 2)
	denominator = (2 * np.pi * bandwidth ** 2) ** 0.5
	return np.exp(numerator) / denominator

def triggering_kernel(alpha, reference_time, time_intervals, bandwidth):
	"""
	Compute triggering kernel for Hawkes process.
	
	This function computes the temporal triggering intensity for a Hawkes process
	by combining multiple RBF kernels with learned weights (alpha parameters).
	The triggering kernel models how past events influence the intensity of
	future events through temporal decay.
	
	The triggering kernel is defined as:
	λ(t) = Σᵢ αᵢ * Kᵢ(t)
	
	where Kᵢ(t) are RBF kernels and αᵢ are their weights.
	
	Args:
		alpha (np.array): Weights for RBF kernels (must be positive)
		reference_time (np.array): Reference times for RBF kernels
		time_intervals (float/np.array): Time differences from past events
		bandwidth (np.array): Bandwidth parameters for RBF kernels
		
	Returns:
		np.array: Triggering intensity values
	"""
	# Reshape time intervals for broadcasting with reference times
	time_intervals = time_intervals.reshape(-1, 1)
	
	# Handle different alpha dimensions (for batch processing)
	if len(alpha.shape) == 3:
		# Batch processing case: alpha has shape (batch, time, features)
		return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis=1), axis=1)
	else:
		# Single case: alpha has shape (features,)
		return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis=0), axis=0)

def g_theta(timeseq, reference_time, bandwidth, max_time):
	"""
	Compute g_theta function for Pitman-Yor Hawkes Process (PYHP).
	
	This function computes the integral of RBF kernels over the time interval
	[0, max_time] for a given time sequence. The g_theta function is used in
	the likelihood computation for Bayesian parameter estimation.
	
	The function computes:
	g_θ(t) = ∫₀ᵀ K(t - τ) dτ
	
	where K is the RBF kernel and T is max_time.
	
	Args:
		timeseq (np.array): Time sequence of events in the cluster
		reference_time (np.array): Reference times for RBF kernels
		bandwidth (np.array): Bandwidth parameters for RBF kernels
		max_time (float): Maximum time for integration
		
	Returns:
		np.array: g_theta values for each reference time
	"""
	timeseq = timeseq.reshape(-1, 1)
	
	# Compute complementary error function integrals for RBF kernels
	results = 0.5 * (erfc(-reference_time / (2 * bandwidth ** 2) ** 0.5) - 
					erfc((max_time - timeseq - reference_time) / (2 * bandwidth ** 2) ** 0.5))
	
	return np.sum(results, axis=0)

def update_triggering_kernel(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time, log_priors):
	"""
	Update triggering kernel parameters using Sequential Monte Carlo.
	
	This function performs Bayesian parameter estimation for the triggering kernel
	using importance sampling. It computes the posterior distribution over alpha
	parameters given the observed time sequence and updates the kernel parameters
	accordingly.
	
	The algorithm:
	1. Compute log likelihood for each alpha sample
	2. Combine with log priors to get log posterior weights
	3. Normalize weights and compute weighted average of alpha parameters
	
	Args:
		timeseq (list): Time sequence including current time
		alphas (np.array): Pre-sampled alpha parameters (shape: sample_num, alpha_dim)
		reference_time (np.array): Reference times for RBF kernels
		bandwidth (np.array): Bandwidth parameters for RBF kernels
		base_intensity (float): Base intensity parameter
		max_time (float): Maximum time for computation
		log_priors (np.array): Log prior probabilities for alpha samples
		
	Returns:
		np.array: Updated alpha parameters (weighted average)
	"""
	# Compute log likelihood for each alpha sample
	logLikelihood = log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time)
	
	# Combine log priors and log likelihoods to get log posterior weights
	log_update_weight = log_priors + logLikelihood
	
	# Prevent numerical overflow by subtracting maximum
	log_update_weight = log_update_weight - np.max(log_update_weight)
	
	# Convert to probability space and normalize
	update_weight = np.exp(log_update_weight)
	update_weight = update_weight / np.sum(update_weight)
	update_weight = update_weight.reshape(-1, 1)
	
	# Compute weighted average of alpha parameters
	alpha = np.sum(update_weight * alphas, axis=0)
	return alpha

def log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time):
	"""
	Compute log likelihood for a time sequence in a cluster for SMC.
	
	This function computes the log likelihood of observing a given time sequence
	under the Hawkes process model with different alpha parameter values.
	The likelihood combines the base intensity and triggering effects from
	past events.
	
	The log likelihood is computed as:
	log L = -Λ₀ - Σᵢ αᵢgᵢ(θ) + Σⱼ log λ(tⱼ)
	
	where Λ₀ is the base intensity term, gᵢ(θ) are the g_theta integrals,
	and λ(tⱼ) are the triggering intensities at each event time.
	
	Args:
		timeseq (list): Time sequence including current time
		alphas (np.array): Alpha parameters for each sample (shape: sample_num, alpha_dim)
		reference_time (np.array): Reference times for RBF kernels
		bandwidth (np.array): Bandwidth parameters for RBF kernels
		base_intensity (float): Base intensity parameter
		max_time (float): Maximum time for computation
		
	Returns:
		np.array: Log likelihood values for each alpha sample
	"""
	# Base intensity term
	Lambda_0 = base_intensity * max_time
	
	# Compute g_theta integrals for each alpha sample
	alphas_times_gtheta = np.sum(alphas * g_theta(timeseq, reference_time, bandwidth, max_time), axis=1)
	
	# Validate time sequence length
	if len(timeseq) == 1:
		raise Exception('The length of time sequence must be larger than 1.')
	
	# Compute time intervals between consecutive events
	time_intervals = timeseq[-1] - timeseq[:-1]
	
	# Reshape alphas for broadcasting with time intervals
	alphas = alphas.reshape(-1, 1, alphas.shape[-1])
	
	# Compute triggering intensities at each event time
	triggers = np.log(triggering_kernel(alphas, reference_time, time_intervals, bandwidth))
	
	# Combine all terms to get log likelihood
	return -Lambda_0 - alphas_times_gtheta + triggers

def log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, vocabulary_size, priors):
	"""
	Compute log probability of Dirichlet-multinomial distribution.
	
	This function computes the log probability of observing a document's word
	distribution given a cluster's aggregated word distribution under the
	Dirichlet-multinomial model. This is used for textual similarity computation
	in the Sequential Monte Carlo algorithm.
	
	The Dirichlet-multinomial distribution models the probability of observing
	a document given the cluster's word distribution, incorporating uncertainty
	through Dirichlet priors.
	
	Args:
		cls_word_distribution (np.array): Cluster's aggregated word distribution
		doc_word_distribution (np.array): Document's word distribution
		cls_word_count (int): Total word count in cluster (including document)
		doc_word_count (int): Word count in the document
		vocabulary_size (int): Size of the vocabulary
		priors (np.array): Dirichlet prior parameters
		
	Returns:
		float: Log probability of the document given the cluster
	"""
	priors_sum = np.sum(priors)
	log_prob = 0
	
	# Add log probability terms using gamma functions
	log_prob += gammaln(cls_word_count - doc_word_count + priors_sum)
	log_prob -= gammaln(cls_word_count + priors_sum)
	log_prob += np.sum(gammaln(cls_word_distribution + priors))
	log_prob -= np.sum(gammaln(cls_word_distribution - doc_word_distribution + priors))
	
	return log_prob

# =============================================================================
# TEST FUNCTIONS
# =============================================================================
# The following functions are used for testing and validation of the utility
# functions. They demonstrate proper usage and verify correct implementation.

def test_dirichlet():
	"""
	Test function for Dirichlet distribution sampling.
	
	This function tests the dirichlet() function by sampling multiple
	probability vectors and verifying they sum to 1.
	"""
	alpha = dirichlet(np.array([1]* 10))
	sample_alpha_list = [dirichlet([1]* 10) for _ in range(3000)]
	print('len(sample_alpha_list)',len(sample_alpha_list))
	print(np.sum(sample_alpha_list[0]))

def test_multinomial():
	"""
	Test function for multinomial distribution sampling.
	
	This function tests the multinomial() function by sampling from
	a multinomial distribution with given probabilities.
	"""
	probabilities = dirichlet(np.array([1]* 10))
	result = multinomial(5, probabilities)
	print(result)

def test_EfficientImplementation():
	"""
	Test function for efficient implementation computation.
	
	This function tests the EfficientImplementation() function by
	computing the earliest time for efficient computation.
	"""
	tu = EfficientImplementation(100, [3,7,11], [2,5,10])
	print(tu)

def test_log_Dirichlet_CDF():
	"""
	Test function for Dirichlet CDF computation.
	
	This function tests the log_Dirichlet_CDF() function by
	computing log probabilities for sampled outcomes.
	"""
	prior = np.array([1]*10)
	outcomes = dirichlet(prior)
	print(outcomes)
	print(log_Dirichlet_CDF(outcomes, prior))

def test_RBF_kernel():
	"""
	Test function for RBF kernel computation.
	
	This function tests the RBF_kernel() function by computing
	kernel values for given parameters.
	"""
	refernce_time = np.array([3, 7, 11])
	bandwidth = np.array([5, 5, 5])
	time_intervals = 3
	print(RBF_kernel(refernce_time, time_intervals, bandwidth))
	print(RBF_kernel(11,3,5))

def test_triggering_kernel():
	"""
	Test function for triggering kernel computation.
	
	This function tests the triggering_kernel() function by
	computing triggering intensities for given parameters.
	"""
	reference_time = np.array([3, 7, 11])
	bandwidth = np.array([5, 5, 5])
	time_intervals = np.array([1, 3])
	time_intervals = time_intervals.reshape(-1, 1)
	print(time_intervals.shape)
	alpha = dirichlet([1] * 3)
	print(alpha)
	print(RBF_kernel(reference_time, time_intervals, bandwidth))
	print(triggering_kernel(alpha, reference_time, time_intervals, bandwidth))
	time_intervals = np.array([1, 3, 50])
	print(triggering_kernel(alpha, reference_time, time_intervals, bandwidth))

def test_g_theta():
	"""
	Test function for g_theta computation.
	
	This function tests the g_theta() function by computing
	integrals for a given time sequence.
	"""
	timeseq = np.arange(0.2, 1000000, 0.6)
	bandwidth = np.array([5, 5, 5])
	reference_time = np.array([3, 7, 11])
	current_time = timeseq[-1]
	T = current_time + 1
	output = g_theta(timeseq, reference_time, bandwidth, T)

def test_log_likelihood():
	"""
	Test function for log likelihood computation.
	
	This function tests the log_likelihood() function by computing
	likelihoods for a given time sequence and alpha samples.
	"""
	timeseq = np.arange(0.2, 1000, 0.6)
	alpha0 = np.array([1, 1, 1])
	bandwidth = np.array([5, 5, 5])
	reference_time = np.array([3, 7, 11])
	sample_num = 1000
	current_time = timeseq[-1]
	T = current_time + 1
	base_intensity = 1

	alphas = []
	log_priors = []
	for _ in range(sample_num):
		alpha = dirichlet(alpha0)
		log_prior = log_Dirichlet_CDF(alpha, alpha0)
		alphas.append(alpha)
		log_priors.append(log_prior)

	alphas = np.array(alphas)
	log_priors = np.array(log_priors)

	logLikelihood = log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, T)
	print(logLikelihood)

def test_update_triggering_kernel():
	"""
	Test function for triggering kernel parameter updates.
	
	This function tests the update_triggering_kernel() function by
	updating alpha parameters based on observed time sequences.
	"""
	# Generate parameters
	timeseq = np.arange(0.2, 1000, 0.1)
	alpha0 = np.array([1, 1, 1])
	bandwidth = np.array([5, 5, 5])
	reference_time = np.array([3, 7, 11])
	sample_num = 3000
	base_intensity = 1
	current_time = timeseq[-1]
	T = current_time + 1

	alphas = []
	log_priors = []
	for _ in range(sample_num):
		alpha = dirichlet(alpha0)
		log_prior = log_Dirichlet_CDF(alpha, alpha0)
		alphas.append(alpha)
		log_priors.append(log_prior)

	alphas = np.array(alphas)
	log_priors = np.array(log_priors)

	alpha = update_triggering_kernel(timeseq, alphas, reference_time, bandwidth, base_intensity, T, log_priors)
	print(alpha)

def test_log_dirichlet_multinomial_distribution():
	"""
	Test function for Dirichlet-multinomial distribution computation.
	
	This function tests the log_dirichlet_multinomial_distribution() function
	using real document data loaded from pickle files.
	"""
	with open('./data/meme/meme_docs.pkl', 'rb') as r:
		documents = pickle.load(r)

	cls_word_distribution = documents[0].word_distribution +  documents[1].word_distribution
	doc_word_distribution  = documents[1].word_distribution
	cls_word_count = documents[0].word_count + documents[1].word_count
	doc_word_count = documents[1].word_count
	vocabulary_size = len(documents[0].word_distribution)
	priors = np.array([1] * vocabulary_size)
	print('cls_word_count', cls_word_count)
	print('doc_word_count', doc_word_count)
	logprob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, vocabulary_size, priors)
	print(logprob)




class ClusterWordAnalyzer(object):
	"""
	Analyze and visualize word distributions at the cluster level.

	This class extracts word→frequency mappings from a given particle's clusters,
	persists them (JSON/CSV), and renders circular word clouds for publication.

	Attributes:
		particle (object): Object with attribute `clusters`; each cluster exposes
		                   a 1D array-like `word_distribution` of nonnegative counts.
		id2word (dict): Mapping from vocabulary IDs (str or int) to tokens (str).
	"""

	def __init__(self, particle, id2word):
		"""
		Initialize the analyzer.

		Args:
			particle (object): Particle containing `clusters`.
			id2word (dict): Vocabulary ID → token mapping. Keys may be str or int.
		"""
		super(ClusterWordAnalyzer, self).__init__()
		self.particle = particle
		self.id2word = id2word

	def get_word_freq(self, cluster_idx, top_n=None, min_count=1.0, stopwords=None):
		"""
		Construct an ordered word→frequency dictionary for a given cluster.

		Args:
			cluster_idx (int): Zero-based cluster index in `particle.clusters`.
			top_n (int or None): If given, keep only the top_n most frequent terms.
			min_count (float): Discard terms with frequency < min_count.
			stopwords (Iterable[str] or None): Words to exclude after ID→token mapping.

		Returns:
			dict: {word: float(count)} in descending order of frequency.

		Raises:
			IndexError: If `cluster_idx` is out of range.
			ValueError: If the cluster lacks a valid 1D nonnegative `word_distribution`.
		"""
		clusters = getattr(self.particle, 'clusters', None)
		if clusters is None or not (0 <= cluster_idx < len(clusters)):
			raise IndexError('cluster_idx=%d is out of range.' % cluster_idx)

		wd = getattr(clusters[cluster_idx], 'word_distribution', None)
		if wd is None:
			raise ValueError("Selected cluster has no 'word_distribution' attribute.")

		counts = np.asarray(wd, dtype=float)
		if counts.ndim != 1:
			raise ValueError('word_distribution must be a 1D array-like.')
		if np.any(counts < 0):
			raise ValueError('word_distribution contains negative entries.')

		# filter by min_count
		idx = np.flatnonzero(counts >= float(min_count))

		# sort descending
		order = np.argsort(counts[idx])[::-1]
		sorted_idx = idx[order]

		# optional truncation
		if top_n is not None and top_n >= 0:
			sorted_idx = sorted_idx[:top_n]

		# build mapping
		sw = set(stopwords) if stopwords is not None else None
		result = {}
		for i in map(int, sorted_idx):
			token = self._word_from_id(i)
			if token is None:
				token = '<UNK_%d>' % i
			if sw is not None and token in sw:
				continue
			result[token] = float(counts[i])

		return result

	def save_word_freq(self, word_freq, out_json, out_csv=None):
		"""
		Save a word→frequency mapping to JSON (and optionally CSV).

		Args:
			word_freq (dict): Mapping {word: count}, e.g., output of `get_word_freq`.
			out_json (str): Destination path for JSON file.
			out_csv (str or None): Destination path for CSV file (optional).
		"""
		out_json_path = Path(out_json)
		out_json_path.parent.mkdir(parents=True, exist_ok=True)
		with out_json_path.open('w', encoding='utf-8') as f:
			json.dump(dict(word_freq), f, ensure_ascii=False, indent=2)

		if out_csv:
			out_csv_path = Path(out_csv)
			out_csv_path.parent.mkdir(parents=True, exist_ok=True)
			with out_csv_path.open('w', newline='', encoding='utf-8') as f:
				writer = csv.writer(f)
				writer.writerow(['word', 'count'])
				for w, c in word_freq.items():
					writer.writerow([w, c])

	def make_wordcloud(self, cluster_idx, out_path, top_n=200, min_count=1.0,
					   stopwords=None, width=1200, height=1200, background_color='white',
					   circular=True, radius_ratio=0.95, random_state=42, color_func=None):
		"""
		Generate and save a high-resolution (optionally circular) word cloud.

		Args:
			cluster_idx (int): Zero-based cluster index.
			out_path (str): Destination PNG path.
			top_n (int): Number of most frequent terms to include.
			min_count (float): Minimum frequency threshold.
			stopwords (Iterable[str] or None): Words to exclude.
			width (int): Output image width (px).
			height (int): Output image height (px).
			background_color (str): Background color (e.g., 'white').
			circular (bool): If True, apply a circular mask.
			radius_ratio (float): Circle radius as a fraction of min(width, height)/2.
			random_state (int or None): Seed for reproducible layout.
			color_func (callable or None): Optional recoloring function.

		Returns:
			None
		"""
		# Guard against missing optional dependency
		if WordCloud is None:
			raise ImportError(
				"wordcloud is not installed. Install it via `pip install wordcloud` "
				"or `conda install -c conda-forge wordcloud`."
			)
		freqs = self.get_word_freq(
			cluster_idx=cluster_idx,
			top_n=top_n,
			min_count=min_count,
			stopwords=stopwords
		)

		mask = self._circular_mask(width, height, radius_ratio) if circular else None

		wc = WordCloud(
			width=width,
			height=height,
			background_color=background_color,
			prefer_horizontal=0.9,
			collocations=False,
			normalize_plurals=False,
			mask=mask,
			random_state=random_state
		).generate_from_frequencies(freqs)

		if color_func is not None:
			wc.recolor(color_func=color_func, random_state=random_state)

		out = Path(out_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		wc.to_file(str(out))

	def _word_from_id(self, idx):
		"""
		Resolve a vocabulary index to a token via `id2word`.

		Tries both integer and string keys to accommodate serialized dictionaries.

		Args:
			idx (int): Vocabulary index.

		Returns:
			str or None: Token if present; otherwise None.
		"""
		if idx in self.id2word:
			return self.id2word[idx]
		key_str = str(idx)
		return self.id2word.get(key_str, None)

	@staticmethod
	def _circular_mask(width, height, radius_ratio):
		"""
		Construct a circular mask array for `WordCloud`.

		Pixels outside the circle are 255 (masked), inside are 0 (valid).

		Args:
			width (int): Mask width in pixels.
			height (int): Mask height in pixels.
			radius_ratio (float): Radius as a fraction of min(width, height)/2.

		Returns:
			np.ndarray: (height × width) uint8 mask suitable for `WordCloud(mask=...)`.

		Raises:
			ValueError: If `radius_ratio` is not in (0, 1].
		"""
		if not (0.0 < radius_ratio <= 1.0):
			raise ValueError('radius_ratio must be in (0, 1].')

		y, x = np.ogrid[:height, :width]
		cx, cy = width / 2.0, height / 2.0
		r = radius_ratio * min(cx, cy)
		mask_bool = (x - cx) ** 2 + (y - cy) ** 2 > r ** 2
		return (mask_bool.astype(np.uint8) * 255)
	
def main():
	"""
	Main function for testing utility functions.
	
	This function serves as the entry point for testing various utility functions.
	It can be used to run individual test functions or demonstrate functionality
	of the utility module. Currently configured to test the Dirichlet-multinomial
	distribution computation.
	"""
	# Uncomment the following lines to run specific tests:
	#test_update_triggering_kernel()
	#test_log_likelihood()

	# Example of active cluster filtering (commented out)
	'''
	tu = 75
	active_clusters = {1:[50, 60, 76, 100], 2:[10,20,30,40], 3:[100,2000]}
	print(active_clusters)

	for cluster_index in active_clusters.keys():
		timeseq = active_clusters[cluster_index]
		active_timeseq = [t for t in timeseq if t > tu]
		if not active_timeseq:
			del active_clusters[cluster_index]
		else:
			active_clusters[cluster_index] = active_timeseq

	print(active_clusters)
	'''




if __name__ == '__main__':
	main()