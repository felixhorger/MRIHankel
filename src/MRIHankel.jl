
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
	@generated function hankel_matrix(
		data::AbstractArray{<: Number, N},
		kernelsize::NTuple{M, Integer}
	) where {N,M}
		@assert M == N - 1

		return quote
			@assert iseven(kernelsize)

			# Get dimensions
			data_shape = size(data)
			kspace_shape = data_shape[1:$M]
			channels = data_shape[$N]
			neighbours = prod(kernelsize)
			neighbours_and_channels = neighbours * channels

			# Select only point for which the convolution kernel does not leave the calibration area
			reduced_shape = kspace_shape .- kernelsize

			# Allocate space
			hankel = Array{ComplexF64, $N}(undef, reduced_shape..., neighbours_and_channels)

			@inbounds @nloops $M k (d -> 1:reduced_shape[d]) begin # Iterate over spatial positions
				# Reset counter for neighbours and channels
				i = 1
				# Get tuple of spatial indices
				k = @ntuple $M d -> k_d
				for $(Symbol("l_$N")) = 1:channels # Sneaky
					@nloops $M l (d -> 0:kernelsize[d]-1) (d -> l_d += k_d) begin # Iterate over neighbours
						# Shift to current position
						hankel[k..., i] = @nref $N data l
						i += 1
					end
				end
			end
			# Flatten spatial dimensions to make it a matrix
			return reshape(hankel, prod(reduced_shape), neighbours_and_channels)
		end
	end

end

