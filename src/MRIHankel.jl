
module MRIHankel
	
	export hankel_matrix

	using Base.Cartesian

	function hankel_matrix(data::AbstractArray{T, N}, kernelsize::NTuple{D, Integer}, neighbours::NTuple{M, CartesianIndex{D}}) where {T<:Number, N, D, M}
		@assert N > 1
		@assert N == D + 1

		# Get dimensions
		data_shape = size(data)
		num_channels = data_shape[N]
		shape = data_shape[1:D]
		num_neighbours_and_channels = M * num_channels

		# Select only points for which the convolution kernel does not leave the data area
		reduced_shape = shape .- kernelsize

		# Allocate space
		hankel = Array{ComplexF64, N}(undef, reduced_shape..., num_neighbours_and_channels)

		one_index_shift = CartesianIndex(ntuple(d -> 1, Val(D))...)
		i = 1 # counter for neighbours and num_channels
		for L in neighbours
			for c = 1:num_channels
				for K in CartesianIndices(reduced_shape) # Iterate over spatial positions
					# Shift to current position
					κ = L + K - one_index_shift
					# Extract from data
					hankel[K, i] = data[L, c]
				end
				i += 1 # Next neighbour
			end
		end
		# Flatten spatial dimensions to make it a matrix
		return reshape(hankel, prod(reduced_shape), num_neighbours_and_channels)
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

	function hankel_matrix(data::AbstractArray{<: Number}, neighbours::NTuple{N, CartesianIndex{D}}) where {N, D}
		kernelsize = Tuple(maximum(I[i] for I in neighbours) for i = 1:D)
		return hankel_matrix(data, kernelsize, neighbours)
	end
end

