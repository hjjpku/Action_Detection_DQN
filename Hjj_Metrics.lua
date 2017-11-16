max_dist = 1000000

function func_calculate_overlapping(mask1, mask2)
	-- mask2 is ground truth
	local beg_i = 0
	local last_i = 0

	if mask1[1] > mask2[1] then
		beg_i = mask1[1]
	else
		beg_i = mask2[1]
	end
	
	if mask1[2] > mask2[2] then
		last_i = mask2[2]
	else
		last_i = mask1[2]
	end
	
	if beg_i > last_i then
		return 0
	else
		return (last_i - beg_i + 1)/(mask2[2] - mask2[1] + 1)
	end
end

function func_calculate_iou(mask1, mask2)
	local beg_i = 0
	local last_i = 0
	local beg_u = 0
	local last_u = 0
	if mask1[1] > mask2[1] then
		beg_i = mask1[1]
		beg_u = mask2[1]
	else
		beg_i = mask2[1]
		beg_u = mask1[1]
	end
	
	if mask1[2] > mask2[2] then
		last_i = mask2[2]
		last_u = mask1[2]
	else
		last_i = mask1[2]
		last_u = mask2[2]
	end
	
	if beg_i > last_i then
		return 0
	else
		return (last_i - beg_i)/(last_u - beg_u)
	end
end


function func_follow_iou(mask, gt_mask, available_objects, iou_table)
	local result_table = torch.Tensor(iou_table:size()):fill(0)
	local iou = 0
	local new_iou = 0
	local index = 0
	
	for i = 1, table.getn(gt_mask)
	do
		if available_objects[i] == 1 then
			iou = func_calculate_iou(mask, gt_mask[i])
			result_table[i] = iou
		else
			result_table[i] = -1
		end
	end
	new_iou, index = torch.max(result_table, 1)
	
	-- from tensor to numeric type
	new_iou = new_iou[1]
	index = index[1]
	
	iou = iou_table[index]
	
	return iou, new_iou, result_table, index
end

function func_calculate_dist(mask1, mask2)
	local dist = max_dist
	if mask1[1] >= mask2[2] then
		dist = mask1[1] - mask2[2] + 1
	elseif mask1[2] <= mask2[1] then
		dist = mask2[1] - mask1[2]
	else
		dist = 0
	end
	return dist
end

function func_follow_dist_iou(mask, gt_mask, available_objects, iou_table,dist_table)
	local result_iou_table = torch.Tensor(iou_table:size()):fill(0)
	local result_dist_table = torch.Tensor(dist_table:size()):fill(0)
	local iou = 0
	local new_iou = 0
	local dist = max_dist
	local old_dist = max_dist
	local index = 0
	
	for i = 1, table.getn(gt_mask)
	do
		if available_objects[i] == 1 then
			iou = func_calculate_iou(mask, gt_mask[i])
			result_iou_table[i] = iou
			if iou == 0 then
				dist = func_calculate_dist(mask, gt_mask[i])
				result_dist_table[i] = dist
			else
				result_dist_table[i] = 0
			end
		else
			result_iou_table[i] = -1
			result_dist_table[i] = max_dist+1
		end
	end
	new_iou, index = torch.max(result_iou_table, 1)
	-- from tensor to numeric type
	new_iou = new_iou[1]
	index = index[1]
	
	if new_iou == 0 then
		new_dist, index = torch.min(result_dist_table, 1)
		new_dist = new_dist[1]
		index = index[1]
	else
		new_dist = 0
	end
	
	iou = iou_table[index]
	dist = dist_table[index]
	
	return dist, new_dist, result_dist_table,
		 			iou, new_iou, result_iou_table, index
	
end


function func_find_max_iou(mask, gt_mask)
	local result_table = torch.Tensor(#gt_mask):fill(0)
	local iou = 0
	local index = 0
	
	for i = 1, table.getn(gt_mask)
	do
		iou = func_calculate_iou(mask, gt_mask[i])
		result_table[i] = iou
	end
	iou, index = torch.max(result_table, 1)
	
	-- from tensor to numeric type
	index = index[1]
	iou = iou[1]
	
	if iou == 0 then -- if iou = zero, map to no gt
		index = 0
	end
	
	return iou, index
end










