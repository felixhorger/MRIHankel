
module MRIHankel
	
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
		kernelsize::Integer # TODO: This could be tuple
	) where N
		@assert N > 1
		M = N - 1

		return quote
			@assert iseven(kernelsize)

			# Get dimensions
			data_shape = size(data)
			kspace_shape = data_shape[1:$M]
			channels = data_shape[$N]
			neighbours = kernelsize^$M
			neighbours_and_channels = neighbours * channels
			neighbour_shift = kernelsize ÷ 2
			neighbour_range = -neighbour_shift:neighbour_shift-1

			hankel = Array{ComplexF64, $N}(undef, kspace_shape..., neighbours_and_channels ) 

			@inbounds @nloops $M k data begin # Iterate over spatial positions
				# Reset counter for neighbours and channels
				i = 1 
				# Get tuple of spatial indices
				k = @ntuple $M d -> k_d
				for $(Symbol("l_$N")) = 1:channels # Sneaky
					@nloops $M l (d -> neighbour_range) (d -> l_d += k_d) begin # Iterate over neighbours
						# Shift to current position
						l = @ntuple $M d -> l_d + k_d
						# Check if outside of array
						if (@nany $M d -> l_d < 1) || (@nany $M d -> l_d > kspace_shape[d])
							hankel[k..., i] = 0 # Outside
						else
							hankel[k..., i] = @nref $N data l
						end
						i += 1
					end
				end
			end
			# Flatten spatial dimensions to make it a matrix
			return reshape(hankel, prod(kspace_shape), neighbours_and_channels)
		end
	end

end

