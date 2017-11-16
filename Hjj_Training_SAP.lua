-- do not mask when extract C3D
-- do not split ground truth
-- no dist 
-- jump
-- no expert
-- for thomas
-- no narrow when 16
-- do not stop when trigger
-- do not mask when extract C3D
-- do not split ground truth
-- no dist 
-- jump
-- no expert
-- for thomas
-- no narrow when 16
-- do not stop for any trigger
-- single side expand
-- jump to where near gt
-- dynamic pooling
-- regression network
require 'Hjj_Read_Input_Cmd'
require 'Hjj_Reinforcement3'
require 'Zt_Interface_PyramidPooling'
require 'Hjj_Mask_and_Actions'
require 'Hjj_Metrics'
require 'optim'

--read input para
local cmd = torch.CmdLine()
opt = func_read_training_cmd(cmd, arg)

-- create log file
local log_file = io.open(opt.log_log, 'w')
if not log_file then
	print("open log file error")
	error("open log file error")
end

-- read training clip id from files
local training_file = './' .. opt.data_path .. '/Thumos_trainlist_new.t7'
print(training_file)
local clip_table = torch.load(training_file)
local tt = clip_table[opt.class]
if tt == nil then
	error('no trainlist file')
end

-- thomas
local training_clip_table={}
training_clip_table = tt
--for i=1,10 do
--	table.insert(training_clip_table, tt[#tt-10+i])
--end

--set training parameters
local max_epochs = opt.epochs
local batch_size = opt.batch_size
local max_steps = 25

--DQN training trick parameters
local experience_replay_buffer_size = opt.replay_buffer
local gamma = 0.90 --discount factor
local epsilon = 1 -- greedy policy
local trigger_thd = 0.5 -- threshold for terminal

local count_train = torch.Tensor(1):fill(0)
local train_period = torch.floor(opt.batch_size/100)


-- number_of_actions and history_action_buffer_size are globle variables in Hjj_Reinforcement
local history_vector_size = number_of_actions * history_action_buffer_size
local input_vector_size = history_vector_size + C3D_size
-- define the last action as trigger
local trigger_action = number_of_actions
local jump_action = number_of_actions-1
local act_alpha = opt.alpha

--init replay memory
local replay_memory = {}

--init reward
local reward = 0
--if opt.model_name is '0', then init DQN model
--else load a saved DQN
local dqn = {}
dqn= func_get_dqn(opt.model_name, log_file)
--load RGN
local rgn = {}
rgn = func_get_rgn('0', log_file)


local fc6 = torch.load('fc6.t7')
fc6:evaluate()

-- set gpu
opt.gpu = func_set_gpu(opt.gpu, log_file)

if opt.gpu >=0 then 
	dqn = dqn:cuda() 
	fc6 = fc6:cuda()
	rgn = rgn:cuda()
end

local params, gradParams = dqn:getParameters()
local params_rgn, gradParams_rgn = rgn:getParameters()

-- define loss function and trainer
local criterion = nn.SmoothL1Criterion()
if opt.gpu >=0 then crirerion = criterion:cuda() end

--defines loss function and trainer for rgn
local criterion_rgn = nn.AbsCriterion()
if opt.gpu >=0 then crirerion_rgn = criterion_rgn:cuda() end

-- training with optim
-- optim paras for optim
local optimState = {learningRate = opt.lr, maxIteration = 1, learningRateDecay = 0.00005, evalCounter = 0}
local optimState_rgn = {learningRate = 1e-4, maxIteration = 1,learningRateDecay = 0.00009, evalCounter = 0}
local logger = optim.Logger(opt.log_err)
logger:setNames{'Training_error', 'epoch'}
local rgn_err = opt.log_err .. '_rgn'
local logger_rgn=optim.Logger(rgn_err)
logger_rgn:setNames{'Training_error'}


--read dataset
local gt_table = func_get_data_set_info(opt.data_path, opt.class, 1)
print(gt_table)

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
--gt_table = func_modify_gt(gt_table, max_gt_length)

-- load C3D model
local C3D_m = torch.load('c3d.t7');
C3D_m:evaluate()

for i = 1, max_epochs
do
	log_file:write('It is the ' .. i .. ' epoch\n')
	print('It is the ' .. i .. ' epoch')
	
	for j, v in pairs(training_clip_table)
	do
		local masked = false 
		local masked_segs={}
		local not_finished = true
		local tmp_gt = gt_table[v]
		local total_frms = tmp_gt[1][3]
		local gt_num = table.getn(tmp_gt)
		local available_objects = torch.Tensor(gt_num):fill(1)
		print('load images')
		local clip_img = func_load_clip(opt.data_path, opt.class, 1, v,total_frms)
		
		log_file:write('\tIt is the ' .. j .. ' clip, clip_id = ' .. 
						v .. ' total_frms = '.. total_frms .. '\n')
		print('\tIt is the '.. j .. ' clip, clip_id = ' .. 
						v .. ' total_frms = '.. total_frms)
		
		local lp_t = torch.round(total_frms * 10/max_steps/avg_len)-- loop times
		if lp_t <= 1 then lp_t = 2 end
		for k = 1, lp_t
		do
			log_file:write('\t\tIt is the ' .. k .. ' gt, from '.. '\n')
			print('\t\tIt is the ' .. k .. ' gt, from '.. '\n')
			
			-- init mask, return beg index and end index of mask
			-- in hjj_mask_and_action
			local cur_mask = func_mask_random_init(total_frms, masked_segs, avg_len)
			local old_mask = cur_mask
			
			-- iou_table record the iou of each gt and cur_mask
			-- reset iou_table in the beginning of each loop
			local iou_table = torch.Tensor(gt_num):fill(0)

			local old_iou = 0
			local new_iou = 0
			local overlap = 0

			local new_dist = max_dist
			

			-- check if available objects left
			if torch.nonzero(available_objects):numel() == 0 then
				not_finished = false
			end
			
			-- calculate iou for cur_mask and gt
			old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
			overlap = func_calculate_overlapping(tmp_gt[index], cur_mask) -- intersec/cur_mask

			

			local now_target_gt = tmp_gt[index]
			
			-- init history action buffer
			local history_vector = torch.Tensor(history_vector_size):fill(0)
			print('\t\t\tInit mask: ' .. cur_mask[1] .. '\t' .. cur_mask[2] .. '\t' .. total_frms..'\n')			
			-- get C3D

			--local C3D_vector = func_get_C3D(opt.data_path, opt.class, 1,
			--								 v, cur_mask[1], cur_mask[2], C3D_m, {}, 27, fc6)
			local C3D_vector = func_get_C3D(clip_img[{ {cur_mask[1], cur_mask[2]},{} }], C3D_m,{},27,fc6)
	
			local input_vector = torch.cat(C3D_vector, history_vector, 1)
			if opt.gpu >=0 then input_vector = input_vector:cuda() end
			
			local bingo = false -- it is a right trigger action or not
			local action = 0 -- init action
			local step_count = 0 -- reset step_count
			reward = 0 -- re-init reward
			--while (not bingo) and (step_count < max_steps) and not_finished
			while (step_count < max_steps) and not_finished
			do
				log_file:write('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							 .. overlap .. '\n')
				print('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlapt = '
							  .. overlap .. '\n')
				-- run DQN	
				local action_output = dqn:forward(input_vector)
				print(action_output)
				local tmp_flag = 0
				local trigger_memory = {}
				
				-- It is checking for last non-trigger action, which may actually lead to an 
				-- terminal state; we force it to be terminal action in case actual IoU 
				-- is higher than 0.5, to train faster the agent; 
				local tmp_v = 0
				tmp_v, action = torch.max(action_output,1)
				action = action[1]-- from tensor to numeric type
				
				if (cur_mask[2]-cur_mask[1]+1) >= max_gt_length*2 and (action == 4 or action == 5) then
					-- forbid expand than max_gt_length
					-- choose a random action
					action = torch.random(torch.Generator(),1,3)
				elseif (cur_mask[2]-cur_mask[1]) <= 16 and action == 3 then
					action = torch.random(torch.Generator(),1,4)
					if action == 3 then action = 5 end
				end
				if action == trigger_action then 
					tmp_flag = 1 
				elseif i < max_epochs and new_iou > trigger_thd then
					--action = trigger_action
					tmp_flag = 2
				elseif i < max_epochs and new_iou == 0 then
					action = jump_action
				elseif torch.uniform(torch.Generator()) < epsilon then -- greedy policy
					action = torch.random(torch.Generator(),1,number_of_actions)
				end
				
				local localize_reg = torch.Tensor(2):fill(0)

				if action == trigger_action then -- estemated as trigger
					old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
					overlap = func_calculate_overlapping(tmp_gt[index], cur_mask)

					now_target_gt = tmp_gt[index]

					reward = func_get_reward_trigger(new_iou)

					if reward > 0 then
						localize_reg[1] = (now_target_gt[1]-cur_mask[1])/(cur_mask[2]-cur_mask[1]+1)
						localize_reg[2] = (now_target_gt[2]-cur_mask[2])/(cur_mask[2]-cur_mask[1]+1)	
					end

					step_count = step_count+1
					bingo = true
					
					log_file:write('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							 .. overlap .. '; self = '.. tmp_flag .. '\n')
					print('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							  .. overlap .. '; self = '.. tmp_flag ..'\n')
					trigger_memory[1]  = {input_vector, number_of_actions, reward, input_vector, localize_reg}
					action = torch.random(torch.Generator(),1,number_of_actions-1)
					--****************************
					--step_count = max_steps -- to jump out of the loop
					--****************************
				elseif tmp_flag == 2 then
					-- forced trigger
					old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
					overlap = func_calculate_overlapping(tmp_gt[index], cur_mask)

					now_target_gt = tmp_gt[index]

					reward = func_get_reward_trigger(new_iou)

					if reward > 0 then
						localize_reg[1] = (now_target_gt[1]-cur_mask[1])/(cur_mask[2]-cur_mask[1]+1)
						localize_reg[2] = (now_target_gt[2]-cur_mask[2])/(cur_mask[2]-cur_mask[1]+1)	
					end

					step_count = step_count+1
					
					log_file:write('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. number_of_actions ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							 .. overlap .. '; self = '.. tmp_flag .. '\n')
					print('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. number_of_actions ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							  .. overlap .. '; self = '.. tmp_flag ..'\n')
							  
					-- add to memory					
					trigger_memory[1]  = {input_vector, number_of_actions, reward, input_vector, localize_reg}
				end
				if action == jump_action then
					-- encourage jump action if it is a iou==0 state
					if new_iou <= 0.05 then
						reward = func_get_reward_movement(0, 1,0,0)*0.5 -- half reward
					else
						reward = func_get_reward_movement(1,0,0,0)*(5)
					end
					cur_mask = func_take_advance_action(cur_mask, action, total_frms, act_alpha, tmp_gt)

					old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
					overlap = func_calculate_overlapping(tmp_gt[index], cur_mask)
					now_target_gt = tmp_gt[index]
					old_iou = new_iou

					history_vector = func_update_history_vector(history_vector, action)
					step_count = step_count + 1
				elseif action ~= trigger_action then -- take action
				-- 1 move forward; 2 move back; 3 narrow; 4 left_expand; 5 right_expand 
					cur_mask = func_take_advance_action(cur_mask, action, total_frms, act_alpha,tmp_gt)
					
					old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
					overlap = func_calculate_overlapping(tmp_gt[index], cur_mask)
					now_target_gt = tmp_gt[index]
					reward = func_get_reward_movement(old_iou, new_iou,0,0)
					old_iou = new_iou
				
					history_vector = func_update_history_vector(history_vector, action)
					step_count = step_count + 1
					-- log wiil be written at the beginning of the next loop
				end
				--local C3D_vector = func_get_C3D(opt.data_path, opt.class, 1,
				--							 v, cur_mask[1], cur_mask[2], C3D_m, {}, 27, fc6)
				local C3D_vector = func_get_C3D(clip_img[{ {cur_mask[1], cur_mask[2]},{} }], C3D_m,{},27,fc6)
				local new_input_vector = torch.cat(C3D_vector, history_vector, 1)
				if opt.gpu >=0 then new_input_vector = new_input_vector:cuda() end
				
				count_train[1] = count_train[1]+1
				-- experience replay
				local tmp_experience  = {input_vector, action, reward, new_input_vector, localize_reg}
				if table.getn(replay_memory) < experience_replay_buffer_size then
					table.insert(replay_memory, tmp_experience)
					if #trigger_memory > 0 then
						if #trigger_memory == 1 then
							table.insert(replay_memory, trigger_memory[1])
						else
							error('wrong trigger memory')
						end
					end
					input_vector = new_input_vector
				else
					-- replay_memory is a stack
					table.remove(replay_memory, 1)
					table.insert(replay_memory, tmp_experience)
					if #trigger_memory > 0 then
						table.remove(replay_memory, 1)
						table.insert(replay_memory, trigger_memory[1])
					end
					
					local tmp_mod = torch.fmod(count_train,train_period)
					tmp_mod = tmp_mod[1]
					if tmp_mod == 0 then

						local minibatch = func_sample(replay_memory, batch_size) -- in Hjj_Reinforcement
						local memory = {}
						-- construct training set
						local training_set = {data=torch.Tensor(batch_size, input_vector_size),
												 label=torch.Tensor(batch_size, number_of_actions)}
						function training_set:size() return batch_size end
						setmetatable(training_set, {__index = function(t,i) 
															return {t.data[i], t.label[i]} end})
					
						if opt.gpu >= 0 then 
							training_set.data = training_set.data:cuda()
							training_set.label = training_set.label:cuda() 
						end
					
						-- construct training set for rgn
						local count_batch = 0
						for l,memory in pairs(minibatch)
						do
							local tmp_action = memory[2]
							local tmp_reward = memory[3]
							if tmp_action == trigger_action and tmp_reward > 0 then
								count_batch = count_batch+1
							end
						end	
						local training_set_rgn = {}					
						if count_batch > 0 then
							training_set_rgn = {data=torch.Tensor(count_batch, input_vector_size),
												 label=torch.Tensor(count_batch, 2)}
							function training_set_rgn:size() return count_batch end
							setmetatable(training_set_rgn, {__index = function(t,i) 
															return {t.data[i], t.label[i]} end})
						end
						if opt.gpu >= 0 then 
							training_set_rgn.data = training_set_rgn.data:cuda() 
							training_set_rgn.label = training_set_rgn.label:cuda()
						end

						log_file:write('\t\t\t\t Doing memory replay...\n')
						print('\t\t\t\t Doing memory replay...\n')
						local counter_batch = 1
						for l, memory in pairs(minibatch)
						do
							local tmp_input_vector = memory[1]
							local tmp_action = memory[2]
							local tmp_reward = memory[3]
							local tmp_new_input_vector = memory[4]
							if tmp_action == trigger_action and tmp_reward > 0 then
								if (count_batch-counter_batch < 0) then error('rgn data set error') end
								training_set_rgn.data[counter_batch] = tmp_input_vector
								training_set_rgn.label[counter_batch] = memory[5]
								counter_batch = counter_batch + 1
							end
							local old_action_output = dqn:forward(tmp_input_vector)
							local new_action_output = dqn:forward(tmp_new_input_vector)
							local tmp_v = 0
							local tmp_index = 0
							local y = old_action_output:clone()
							tmp_v, tmp_index = torch.max(new_action_output, 1)
							tmp_v = tmp_v[1]
							tmp_index = tmp_index[1]
							local update_reward = 0
							if (tmp_action == trigger_action) or (tmp_action == jump_action) then
								update_reward = tmp_reward
							else
								update_reward = tmp_reward + gamma * tmp_v
							end
							y[tmp_action] = update_reward
							training_set.data[l] = tmp_input_vector
							training_set.label[l] = y
						end
						-- training
						log_file:write('\t\t\t\t Training...\n')
						print('\t\t\t\t Training...\n')
						local function feval(x)
                            if x ~= params then
                                params:copy(x)
                            end
							gradParams:zero()
							--print(params:sum())
                            
							local outputs = dqn:forward(training_set.data)
                            --print(outputs:sum())
							local loss = criterion:forward(outputs, training_set.label)
							local dloss_doutputs = criterion:backward(outputs, training_set.label)
							--print(dloss_doutputs:sum())
                            --io.read()
                            dqn:backward(training_set.data, dloss_doutputs)
							logger:add{loss*100, i}
							return loss, gradParams
						end
						optim.sgd(feval, params, optimState)
						

						--training rgn
						if count_batch > 0 then
							log_file:write('\t\t\t\t Training RGN...\n')
							print('\t\t\t\t Training RGN...\n')
							local function feval_rgn(x)
	                            if x ~= params_rgn then
	                                params_rgn:copy(x)
	                            end
								gradParams_rgn:zero()
								--print(params:sum())
	                            
								local outputs = rgn:forward(training_set_rgn.data)
	                            --print(outputs:sum())
								local loss = criterion_rgn:forward(outputs, training_set_rgn.label)
								local dloss_doutputs = criterion_rgn:backward(outputs, training_set_rgn.label)
								--print(dloss_doutputs:sum())
	                            --io.read()
	                            rgn:backward(training_set_rgn.data, dloss_doutputs)
								logger_rgn:add{loss}
								return loss, gradParams_rgn
							end
							optim.sgd(feval_rgn, params_rgn, optimState_rgn)

							local tmp_a = torch.Tensor(1)
							local tmp_b = 1000
							tmp_a[1] = optimState_rgn.evalCounter
							local tmp_mod = torch.fmod(tmp_a,tmp_b)
							tmp_mod = tmp_mod[1]
								-- save enviroments
								if tmp_mod == 0 or optimState_rgn.evalCounter == 1 then
									local mdl_name={}
									if opt.gpu >= 0 then
										mdl_name = './model/rgn_pooling_'.. opt.name .. opt.class .. '_'.. optimState_rgn.evalCounter
									else 
										mdl_name = './model/c_'.. opt.name .. opt.class .. '_'.. i
									end
									torch.save(mdl_name, {rgn = rgn})
								end
						end
					end -- mod
					input_vector = new_input_vector
				end -- if memory replay
				
				if action == trigger_action then
					bingo = true
					masked = true
					if reward == 3 then
						table.insert(masked_segs, {cur_mask[1]+torch.floor((cur_mask[2]-cur_mask[1]+1)*0.1),
									cur_mask[2]-torch.floor((cur_mask[2]-cur_mask[1]+1)*0.1)})
					end
				else
					masked = false
				end
			end -- while (not bingo) and (step_count < max_steps) and not_finished
			-- available_objects[index] = 0
		end -- gts loop

	end -- clips loop
	if epsilon > 0.1 then
		epsilon = epsilon - 0.1
	end
	-- save enviroments
	if table.getn(replay_memory) >= experience_replay_buffer_size then
		local mdl_name={}
		if opt.gpu >= 0 then
			mdl_name = './model/g_'.. opt.name .. opt.class .. '_'.. i
		else 
			mdl_name = './model/c_'.. opt.name .. opt.class .. '_'.. i
		end
		torch.save(mdl_name, {dqn = dqn, gpu = opt.gpu})
	end

end -- epochs loop

log_file:close()




















