require 'torch'
require 'math'

local max_mask = 256

function func_mask_random_init(...)
	local arg={...}
	local total_frms = arg[1] 
	-- generate a ramdom mask longger than 16 frames
	-- because C3D need at least 16 frames
	local n1 = 0
	local n2 = 0
	local avg_len = arg[3]
	local generator = torch.Generator()
	if total_frms - torch.floor(avg_len) <= 5 then
		return {1, total_frms}
	else
		n1 = torch.random(generator,1,total_frms - torch.floor(avg_len) )
		n2 = n1 + torch.floor(avg_len)
	end

		-- do not see masked area
		local mask_table = arg[2]
		
		local flag = true
		local count = 0
		while flag and count < 10
		do
			flag = false
			for i=1,#mask_table 
			do
				if (n1 > mask_table[i][1] or n1 < mask_table[i][2]) or 
						(n2 > mask_table[i][1] or n2 < mask_table[i][2]) then
					n1 = torch.random(generator,1,total_frms- torch.floor(max_mask/4))
					n2 = n1+ torch.floor(avg_len)
					flag = true
					break
				end
			end
			count = count+1
		end -- while flag
		return {n1, n2}

end


function func_take_action(old_mask, action, total_frms,alpha, gt)
	
	local new_mask = {}
	local len = 0	
	local offset = 0
	
	if action == 1 then -- move forward
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
			new_mask[2] = old_mask[2] - offset
		else
			new_mask[1] = 1
			new_mask[2] = len
		end 
		
	elseif action == 2 then -- move back
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end 
		
	elseif action == 4 then -- expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha / 2)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
		else
			new_mask[1] = 1
		end 
		
		if (old_mask[2] + offset) < total_frms  then
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[2] = total_frms
		end
		
	elseif action == 3	then -- narrow
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha / 2)
		
		-- check if at least has 16 frames
		if (len - 2*offset) >= 17 then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		else
			offset = torch.floor((len - 16) / 2)
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		end
	elseif action == 5	then -- jump	
		local generator = torch.Generator()
		local num = torch.random(generator,1,#gt)
		local beg = gt[num][1]
		local ed = gt[num][2]
		local ll = ed-beg
		if beg-(old_mask[2]-old_mask[1])+0.1*ll > 1 then 
			beg = torch.floor(beg-(old_mask[2]-old_mask[1])+0.1*ll) 
		else 
			beg = 1 
		end
		ed = torch.floor(ed-0.1*ll)   
		new_mask[1] = torch.random(generator,beg,ed)
		new_mask[2] = new_mask[1]+(old_mask[2]-old_mask[1])
		if new_mask[2] >= total_frms then 
			new_mask[2] = total_frms-1
			new_mask[1]	= new_mask[1] + total_frms - new_mask[2] - 1
		end
	else
		error('Wrong action')
	end 
	if (new_mask[2] - new_mask[1]+1) < 16 then
			print(new_mask[1], new_mask[2])
			error('inadequate frames action' .. action)
	end
	return new_mask
end

-- single side expand
function func_take_advance_action(old_mask, action, total_frms,alpha,gt)
	
	local new_mask = {}
	local len = 0	
	local offset = 0
	
	if action == 1 then -- move forward
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
			new_mask[2] = old_mask[2] - offset
		else
			new_mask[1] = 1
			new_mask[2] = len
		end 
		
	elseif action == 2 then -- move back
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end 
		
	elseif action == 3	then -- narrow
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha / 2)
		
		-- check if at least has 16 frames
		if (len - 2*offset) >= 17 then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		else
			offset = torch.floor((len - 16) / 2)
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		end
	elseif action == 4 then -- left_expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
		else
			new_mask[1] = 1
		end 
		new_mask[2] = old_mask[2]
		
	elseif action == 5 then -- right_expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha)
		
		new_mask[1] = old_mask[1]

		if (old_mask[2] + offset) < total_frms  then
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[2] = total_frms
		end
	elseif action == 6 then -- jump	
--[[		
		local generator = torch.Generator()
		new_mask[1] = torch.random(generator,1,total_frms - (old_mask[2]-old_mask[1]) )
		new_mask[2] = new_mask[1]+(old_mask[2]-old_mask[1])
]]
		local generator = torch.Generator()
		local num = torch.random(generator,1,#gt)
		local beg = gt[num][1]
		local ed = gt[num][2]
		local ll = ed-beg
		if beg-(old_mask[2]-old_mask[1])+0.1*ll > 1 then 
			beg = torch.floor(beg-(old_mask[2]-old_mask[1])+0.1*ll) 
		else 
			beg = 1 
		end
		ed = torch.floor(ed-0.1*ll)   
		new_mask[1] = torch.random(generator,beg,ed)
		new_mask[2] = new_mask[1]+(old_mask[2]-old_mask[1])
		if new_mask[2] >= total_frms then 
			new_mask[2] = total_frms-1
			new_mask[1]	= new_mask[1] + total_frms - new_mask[2] - 1
		end
		if (new_mask[2] - new_mask[1]+1) <= 17 then
			if new_mask[1] < 17 then
				new_mask[2] = 17 + new_mask[1]
			else
				new_mask[1] = new_mask[2] - 17
			end 
		end
	else
		error('Wrong action')
	end 
	if (new_mask[2] - new_mask[1]+1) < 16 then
			print(new_mask[1], new_mask[2])
			error('inadequate frames action' .. action)
	end
	return new_mask
end


-- single side expand validate
function func_take_advance_action_forward(old_mask, action, total_frms,alpha)
	
	local new_mask = {}
	local len = 0	
	local offset = 0
	
	if action == 1 then -- move forward
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
			new_mask[2] = old_mask[2] - offset
		else
			new_mask[1] = 1
			new_mask[2] = len
		end 
		
	elseif action == 2 then -- move back
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end 
		
	elseif action == 3	then -- narrow
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha / 2)
		
		-- check if at least has 16 frames
		if (len - 2*offset) >= 17 then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		else
			offset = torch.floor((len - 18) / 2)
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		end
	elseif action == 4 then -- left_expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
		else
			new_mask[1] = 1
		end 
		new_mask[2] = old_mask[2]
		
	elseif action == 5 then -- right_expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha)
		
		new_mask[1] = old_mask[1]

		if (old_mask[2] + offset) < total_frms  then
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[2] = total_frms
		end
	elseif action == 6 then -- jump	->take a giant leap instead
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha * 3)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end  
	else
		error('Wrong action')
	end 
	if (new_mask[2] - new_mask[1]+1) <= 16 then
			print(new_mask[1], new_mask[2])
			error('inadequate frames action' .. action)
	end
	return new_mask
end

-- temporary for validate
function func_take_action_forward(old_mask, action, total_frms,alpha)
	
	local new_mask = {}
	local len = 0	
	local offset = 0
	
	if action == 1 then -- move forward
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
			new_mask[2] = old_mask[2] - offset
		else
			new_mask[1] = 1
			new_mask[2] = len
		end 
		
	elseif action == 2 then -- move back
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end 
		
	elseif action == 4 then -- expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha / 2)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
		else
			new_mask[1] = 1
		end 
		
		if (old_mask[2] + offset) < total_frms  then
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[2] = total_frms
		end
		
	elseif action == 3	then -- narrow
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha / 2)
		
		-- check if at least has 16 frames
		if (len - 2*offset) >= 17 then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		else
			offset = torch.floor((len - 17) / 2)
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		end
	elseif action == 5	then -- jump -> take a giant leap instead
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha * 3)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end  
	else
		print(action..'!!!!!!!!!!!!!!!!!!')
		error('Wrong action')
	end 
	if (new_mask[2] - new_mask[1]+1) < 17 then
			print(new_mask[1], new_mask[2])
			error('inadequate frames action' .. action)
	end
	return new_mask
end
