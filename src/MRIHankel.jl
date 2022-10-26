
module MRIHankel
	
	export hankel_matrix

	using Base.Cartesian

	function hankel_matrix(data::AbstractArray{T, N}, neighbours::AbstractVector{<: CartesianIndex{D}}, kernelsize::NTuple{D, Integer}) where {T<:Number, N, D}
		@assert N > 1
		@assert N == D + 1
		@assert check_kernelsize(neighbours, kernelsize)

		# Get dimensions
		data_shape = size(data)
		num_channels = data_shape[1]
		shape = data_shape[2:N]
		num_channels_and_neighbours = num_channels * length(neighbours)

		# Select only points for which the convolution kernel does not leave the data area
		reduced_shape = shape .- kernelsize

		# Allocate space
		hankel = Array{ComplexF64, N}(undef, num_channels_and_neighbours, reduced_shape...)

		origin = one(CartesianIndex{D})
		neighbours = [L - origin for L in neighbours]
		for K in CartesianIndices(reduced_shape) # Iterate over spatial positions
			n = 1 # counter for neighbours and num_channels
			for L in neighbours
				κ = L + K
				@inbounds @views hankel[n:n+num_channels-1, K] = data[:, κ]
				n += num_channels
			end
		end
		# Flatten spatial dimensions to make it a matrix
		return reshape(hankel, num_channels_and_neighbours, prod(reduced_shape))
	end

	"""
		data = [spatial dims..., num_channels/contrast etc.]

		Last dimension assumed num_channels/contrast...

		tuples (channel, voxel, neighbour)

		Matrix structure:
		(1, 1, 1), (1, 1, 2), ... (1, 1, last), (2, 1, 1), ...
		(1, 2, 1), (1, 2, 2), ...

	"""
	function hankel_matrix(data::AbstractArray{<: Number}, kernelsize::NTuple{D, Integer}) where D
		@assert all(iseven, kernelsize) # TODO: Not sure why I did this
		return hankel_matrix(data, kernelsize, CartesianIndices(kernelsize))
	end

	function check_kernelsize(neighbours::AbstractVector{<: CartesianIndex{D}}, kernelsize::NTuple{D, Integer}) where D
		return all(0 < I[d] ≤ kernelsize[d] for I in neighbours for d = 1:D)
	end
end

