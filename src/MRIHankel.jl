
module MRIHankel
	
	export hankel_matrix

	using Base.Cartesian
	
	"""
		Last dimension assumed channels/contrast...

		tuples (channel, voxel, neighbour)

		Matrix structure:
		(1, 1, 1), (1, 1, 2), ... (1, 1, last), (2, 1, 1), ...
		(1, 2, 1), (1, 2, 2), ...

	"""
	function hankel_matrix(
		data::AbstractArray{<: Number, N},
		kernelsize::NTuple{M, Integer}
	) where {N,M}
		@assert M == N - 1
		@assert all(iseven, kernelsize)

		# Get dimensions
		data_shape = size(data)
		kspace_shape = data_shape[1:M]
		channels = data_shape[N]
		neighbours = prod(kernelsize)
		neighbours_and_channels = neighbours * channels

		# Select only point for which the convolution kernel does not leave the calibration area
		reduced_shape = kspace_shape .- kernelsize

		# Allocate space
		hankel = Array{ComplexF64, N}(undef, reduced_shape..., neighbours_and_channels)

		one_index_shift = CartesianIndex(ntuple(d -> 1, Val(M))...)
		for K in CartesianIndices(reduced_shape) # Iterate over spatial positions
			# Reset counter for neighbours and channels
			i = 1
			for c = 1:channels
				for L in CartesianIndices(kernelsize) # Iterate over neighbours
					# Shift to current position
					L += K - one_index_shift
					# Extract from data
					hankel[K, i] = data[L, c]
					# Next neighbour
					i += 1
				end
			end
		end
		# Flatten spatial dimensions to make it a matrix
		return reshape(hankel, prod(reduced_shape), neighbours_and_channels)
	end

end

