-- single side expand
-- localization regression
require 'Hjj_Read_Input_Cmd'
require 'Hjj_Reinforcement3'
require 'Hjj_Mask_and_Actions'
require 'Hjj_Metrics'
require 'Zt_Interface_PyramidPooling'

local cmd = torch.CmdLine()
opt = func_read_validate_rgn_cmd(cmd, arg)

-- create log file
local log_file = io.open(opt.log_log, 'w')
if not log_file then
	print("open log file error")
	error("open log file error")
end

local name  = './data_output/track'.. opt.name .. '.txt'
local track_file = io.open(name, 'w')
if not track_file then
	print("open track file error")
	error("open track file error")
end

name  = './data_output/gt'.. opt.name .. '.txt'
local gt_file = io.open(name, 'w')
if not gt_file then
	print("open gt file error")
	error("open gt file error")
end

local training_file = './' .. opt.data_path .. '/Thumos_trainlist_new.t7'
local clip_table = torch.load(training_file)
local tt = clip_table[opt.class]
if tt == nil then
	error('no trainlist file')
end

-- read validate clips from files
local validate_file = './' .. opt.data_path .. '/Thumos_validatelist_new.t7'
--local validate_file = './' .. opt.data_path .. '/Thumos_trainlist_1to19.t7'
print(validate_file)
local clip_table = torch.load(validate_file)
local validate_clip_table = clip_table[opt.class]
if validate_clip_table == nil then
	error('no trainlist file')
end

-- action parameters
local max_steps = opt.max_steps
local trigger_thd = 0.65 -- threshold for terminal
local trigger_action = number_of_actions
local act_alpha = opt.alpha
local max_trigger = 11
local mask_rate = 0.05 -- 1-2*mask_rate of current mask will not be used anymore

-- number_of_actions and history_action_buffer_size are globle variables in Hjj_Reinforcement
local history_vector_size = number_of_actions * history_action_buffer_size
local input_vector_size = history_vector_size + C3D_size

-- load dqn
if opt.model_name == '0' then
	error('model needed')
end
local dqn={}
local ldldl
dqn= func_get_dqn(opt.model_name, log_file)
dqn:evaluate()
local fc6 = torch.load('fc6.t7')
fc6:evaluate()
local rgn = func_get_rgn(opt.rgn_name, log_file)
rgn:evaluate()

-- set gpu
opt.gpu = func_set_gpu(opt.gpu, log_file)
if opt.gpu >=0 then 
	dqn = dqn:cuda() 
	rgn = rgn:cuda()
	fc6 = fc6:cuda()
end

local gt_table = func_get_data_set_info(opt.data_path, opt.class, 1)

-- get average length
local count = 0
local len = 0
for i,v in pairs(tt) do
	local tmp_table = gt_table[v]
	for j,u in pairs(tmp_table) do
		count = count +1
		len = len + u[2]-u[1]
	end
end
local avg_len = torch.floor(len/count)
if avg_len < 16 then avg=16 end
print(avg_len)

local max_gt_length = 128 -- max length to split gt

-- load C3D model
local C3D_m = torch.load('c3d.t7');
C3D_m:evaluate()

-- used to visualize the action sequence
local iou_record_table = {}
local gt_ind_record_table = {}
local mask_record_table= {}
local max_steps = 15 -- prevent from being trapped in local pit 

for i,v in pairs(validate_clip_table)
do
	--if i>1 then break end

		local tmp_gt = gt_table[v]

		for c=1,#tmp_gt do
			gt_file:write(i .. '\t' .. tmp_gt[c][1] .. '\t' .. tmp_gt[c][2] .. '\n')
		end
	
		local total_frms = tmp_gt[1][3]
		print('load images')
		local clip_img = func_load_clip(opt.data_path, opt.class, 1, v,total_frms)
		local masked_segs={}
		local gt_num = table.getn(tmp_gt)
		local steps = 0

		log_file:write('\tIt is the ' .. i .. ' clip, clip_id = ' .. 
							v .. ' total_frms = '.. total_frms .. '\n')
		print('\tIt is the '.. i .. ' clip, clip_id = ' .. 
							v .. ' total_frms = '.. total_frms)
		local lp=1
		local left_frm = total_frms
		local knocked = 0
		local start_wall = 0
		local last_f = 1					

		while (left_frm > 16) and (knocked < 5)
		do
			local iou = 0
			local index = 0
			--local mask = func_mask_random_init(total_frms, masked_segs)
			-- continue from the end of last mask
			local mask = {last_f, last_f+avg_len}
			if last_f - 16 > 0 then 
				mask[1] = mask[1] - 16
				mask[2] = mask[2] - 16
			 end
			
			if mask[2] >= total_frms then 
				mask[2] = total_frms
				knocked = knocked + 1
			end
			if mask[1] == 1 then start_wall = start_wall + 1 end
			
			local history_vector = torch.Tensor(history_vector_size):fill(0) 
			local bingo = false
			local action = 0
			local last_action = 0 
			local step_count = 0
			local output_record = 0
			while (left_frm > 16) and (knocked < 5) and (not bingo) and (step_count < max_steps)
			do
				iou, index = func_find_max_iou(mask, tmp_gt)
				if not(action == 0) then
					track_file:write(i .. '\t' .. lp .. '\t' .. steps .. '\t' ..
									 iou .. '\t' .. mask[1] .. '\t' .. mask[2] .. '\t' .. action ..'\t'.. output_record[trigger_action] .. '\t' .. v .. '\n')
				end
				print('\t\tstep ' .. steps .. '\t; beg = ' .. mask[1] .. '\t ;end = ' .. mask[2] 
								.. ' ; iou ' .. iou .. '\t' .. action ..'\t'.. action-action .. '\t' .. v .. '\n')
												
				--local C3D_vector = func_get_C3D(opt.data_path, opt.class, 1,
				--							 v, mask[1], mask[2], C3D_m, {}, 27, fc6)
				local C3D_vector = func_get_C3D(clip_img[{ {mask[1], mask[2]},{} }], C3D_m,{},27,fc6)
				local input_vector = torch.cat(C3D_vector, history_vector, 1)
			
				if opt.gpu >=0 then input_vector = input_vector:cuda() end
			
				local action_output = dqn:forward(input_vector)
				local tmp_v = 0
				output_record = action_output:clone()
				tmp_v, action = torch.max(action_output,1)
				action = action[1]-- from tensor to numeric type
				-- give a very small number for getting the second max action
				action_output[action] = -111111111 
				
				print('\t\t\tAction = ' .. action .. '\n')
				
				
				if action == 3 and mask[2]-mask[1] <= 16 then
					tmp_v, action = torch.max(action_output,1)
					action = action[1]-- from tensor to numeric type
				elseif (action == 4 or action == 5) and mask[2]-mask[1]+1 >= 2*max_gt_length then
					tmp_v, action = torch.max(action_output,1)
					action = action[1]-- from tensor to numeric type
					if action == 4 or action == 5 then 
						action_output[action] = -111111111  
						tmp_v, action = torch.max(action_output,1)
						action = action[1]-- from tensor to numeric type
					end
					 
				end
				if action == trigger_action then
					print('############### BOOM! #############'.. mask[1] .. 
						' - ' .. mask[2] .. ' ; ' .. total_frms .. '\n')
					
					-- modify with rgn
					local localize_reg = rgn:forward(input_vector)
					local new_1 = mask[1] + torch.floor(localize_reg[1]*(mask[2]-mask[1]+1))
					local new_2 = mask[2] + torch.floor(localize_reg[2]*(mask[2]-mask[1]+1))
					print('Move frome ( '.. mask[1] .. ', '.. mask[2] ..' ) to ( '.. new_1 .. ', '
						.. new_2 .. ' )\n')
					if new_1 <= 0 then new_1 = 1 end
					if new_2 >= total_frms then new_2 = total_frms end
					if (new_2 - new_1) > 0 then 
						mask[1] = new_1
						mask[2] = new_2
					end
					track_file:write(i .. '\t' .. lp .. '\t' .. steps .. '\t' ..
								 iou .. '\t' .. mask[1] .. '\t' .. mask[2] .. '\t' .. action .. '\t' .. output_record[trigger_action] .. '\t' .. v ..'\n')
					bingo = true
					-- not to mask all the area
					if last_f < mask[2] then
						last_f = mask[2]
						left_frm = total_frms - last_f
					end
					if mask[2] == total_frms then
						knocked = knocked + 1
					end
					mask[1] = mask[1]+torch.floor((mask[2]-mask[1])*mask_rate)
					mask[2] = mask[2]-torch.floor((mask[2]-mask[1])*mask_rate)
					table.insert(masked_segs, mask)
				else
					local rand = 0
					mask = func_take_advance_action_forward(mask, action, total_frms, act_alpha*(1+rand))
					if mask[1] == 1 then 
						start_wall = start_wall + 1 
					else
						start_wall = 0	
					end
					if last_f < mask[2] then
						last_f = mask[2]
						left_frm = total_frms - last_f
					elseif last_f - mask[2] >= 64 or start_wall >= 5 then
						bingo = true
						print('~~~~~~go back too much!!!!')
					end
					if mask[2] == total_frms then
						knocked = knocked + 1
					end
				end
				history_vector = func_update_history_vector(history_vector, action)
				steps = steps + 1
				step_count = step_count + 1
			end	--while
			lp = lp+1
		end -- while
	
end --for clips

gt_file:close()
track_file:close()















