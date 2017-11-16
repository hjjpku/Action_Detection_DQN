--
-- C3D Temporal Pooling
--



-- 1) Install Torch:
--        See http://torch.ch/docs/getting-started.html#_

-- some options

--assert(image_dir:sub(image_dir:len()) ~= '/', 'image_dir should not end with /')

-- load dependencies
require 'cutorch'     -- CUDA tensors
require 'nn'          -- neural network package
require 'cudnn'       -- fast CUDA routines for neural networks
require 'paths'       -- utilities for reading directories 
require 'image'       -- reading/processing images
require 'xlua'        -- for progress bar
require 'math'

function func_get_data_set_info(data_path,class_id,flag)

--#######################################
	--local action_class={'High jump','Cricket','Discus throw','Javelin throw','Paintball','Long jump',
	--                     'Bungee jumping','Triple jump','Shot put','Dodgeball','Hammer throw',
	--                     'Skateboarding','Doing motocross','Starting a campfire','Archery',
	--                     'Playing kickball','Pole vault','Baton twirling','Camel ride','Croquet',
	--                     'Curling','Doing a powerbomb','Hurling','Longboarding','Powerbocking','Rollerblading'}
	--action_class[1] = 'High jump'
	local action_class = {'High jump','Javelin throw','BasketballDunk','Billiards',
					'CricketShot','GolfSwing','Ambiguous','BaseballPitch','CleanAndJerk',
					'CliffDiving','CricketBowling','Diving','FrisbeeCatch',
					'HammerThrow','LongJump','PoleVault','Shotput','SoccerPenalty',
					'TennisSwing','ThrowDiscus','VolleyballSpiking'}
--########################################
    local gt_table={}
    local clip_table={}
    local dir_path={}
    dir_path = data_path ..'/'.. action_class[class_id] ..'/'
    local file_clipnum = io.open(dir_path..'ClipsNum.txt',"r")        --the number of clips in class class_id
    if file_clipnum then
	    local n = file_clipnum:read("*n")                  --read the number of clips in class class_id
	    --print('n='..n)
	    io.close(file_clipnum)
	    if(flag==0) then                                   --test dataset has no groudtruth(not released),so return a empty table
                for i=1,n do
                    local Frame_file_test = io.open(dir_path..i..'/FrameNum.txt',"r")   --the file keep the name of frames of each clips
		    		local FrameNum_test = Frame_file_test:read("*n")
		    		io.close(Frame_file_test)
		    		gt_table[i] = {{0,0,FrameNum}}                    
                end
            print('error return')
			return gt_table
	    elseif(flag==1) then                               --training
		--dir_path = data_path..'/'..'train/'..class_id..'/'
		  for i=1,n do
			gt_path = dir_path..i..'/gt.txt'             --the file which keep the groundtruth of each clip
			--print(gt_path)
			--error()
			local file_gt = io.open(gt_path,"r")
			--local j=1
			if file_gt then
			   local Frame_file = io.open(dir_path..i..'/FrameNum.txt',"r")   --the file keep the name of frames of each clips
			   local FrameNum = Frame_file:read("*n")
			   io.close(Frame_file)
			   --while(true) do
		       local gt = {}
			   local file_gtnum=io.open(dir_path..i..'/NumOfGt.txt',"r")         --the file keep the number of groundtruth of each clips
		       local gtnum = file_gtnum:read("*n")
		       io.close(file_gtnum)
		       clip_table={}
			   for k=1,gtnum do
			        gt={}
		           --local temp=file_gt:read("*n")
		           gt[1]=file_gt:read("*n")
			       gt[2]=file_gt:read("*n")
		           gt[3]=FrameNum
		           clip_table[k]=gt
		     
			   end
		        io.close(file_gt)
			     -- end
			   gt_table[i]=clip_table
			else
			   gt_table[i]={{0,0,0}}
			end

		end
	end
   end
   --print(gt_table) 
   return gt_table
end


function func_get_data(data,cover_id)

	local beg_ind = 1
	local end_ind = data:size(1)
    if(end_ind - beg_ind +1 <16) then
		print('the number of frame is less than 16');
		return
	end

    local Length = 16 
	local new_beg_ind = beg_ind

	if((end_ind - beg_ind+1)%16 == 0) then   --if 16*n frames
		new_beg_ind = beg_ind;
		Length = end_ind -beg_ind+1;
	else                                     --more than 16*n frames
		new_beg_ind = math.random((end_ind -beg_ind+1)%16) + beg_ind
		Length = 16*math.floor((end_ind -beg_ind+1)/16)
	end
	
	input_data = data[{ {new_beg_ind,Length-1+new_beg_ind}, {}}]
   
	return input_data

end

function func_load_clip(data_path,class_id,flag,clip_ind,total_frms)
--#######################################

	local action_class = {'High jump','Javelin throw','BasketballDunk','Billiards',
					'CricketShot','GolfSwing','Ambiguous','BaseballPitch','CleanAndJerk',
					'CliffDiving','CricketBowling','Diving','FrisbeeCatch',
					'HammerThrow','LongJump','PoleVault','Shotput','SoccerPenalty',
					'TennisSwing','ThrowDiscus','VolleyballSpiking'}
--########################################
	
	local Width = 112
	local Height = 112
    local Channels = 3 
	local mean_image = {128, 128, 128}

    local path = data_path..'/'..action_class[class_id]..'/'..tostring(clip_ind)..'/'
    --print(path)
	local data = torch.Tensor(total_frms,Channels,Width,Height)

	for t = 1,total_frms do
	    local cover_flag = false;
	    step = 1
	    local f = io.open(path..tostring(t*step..'.jpg'))
		if f then 
        	io.close(f)
		else 
          	print('jpg file not found!')
            return
        end
		
		local im = {}

		im = image.load(path..tostring(t*step..'.jpg'))                 -- read image
		im = image.scale(im, Width, Height)  -- resize image
		im = im * 255                                 -- change range to 0 and 255
		im = im:index(1,torch.LongTensor{3,2,1})      -- change RGB --> BGR
		  -- subtract mean
		for i=1,3 do
			im[{ i, {}, {} }]:add(-mean_image[i])       --normalization
		end

	    data[t] = im
    end

	return data
end


function func_get_C3D(data,c3d_m,cover_id,layer_extract,fc6_m)
    local last_conv_id = 21
	local model = c3d_m
	local layer_to_extract = layer_extract
	local fc6=fc6_m
	print('get images\n')
	local s_t = os.clock()
	local data_orign = func_get_data(data,cover_id)
	data_orign = data_orign:permute(2,1,3,4)	
	data_orign=data_orign:cuda()
	local e_t = os.clock()
	print('get images end, time consuming: '.. e_t-s_t .. '\n')

	print('get C3D\n')
	s_t = os.clock()
	model:forward(data_orign)
	e_t = os.clock()
	print('get C3D end, time consuming: '.. e_t-s_t .. '\n')

	--print('transfer to CPU\n')
	--s_t = os.clock()
	local feat_conv5 = model.modules[last_conv_id].output
	local feat_cpu = feat_conv5:double()
	--e_t = os.clock()
	--print('transfer to CPU end, time consuming: '.. e_t-s_t .. '\n')	
	
	--print('Pooling\n')
	--s_t = os.clock()
	local feat_pooled = func_pyramidpooling(feat_cpu)
	feat_pooled = feat_pooled:cuda()
	--e_t = os.clock()
	--print('Pooling end, time consuming: '.. e_t-s_t .. '\n')	
	print('fc6\n')
	s_t = os.clock()	
	local output = fc6:forward(feat_pooled)
	e_t = os.clock()
	print('fc6 end, time consuming: '.. e_t-s_t .. '\n')	
	local fea_fc6 = fc6.modules[layer_to_extract - last_conv_id].output
    fea_fc6 = fea_fc6:double()

	return fea_fc6
end

function func_pyramidpooling(fea)
	local dim = #fea
	local n = dim[3]
	local length = dim[2]
	local i
	local b
	local width
	local height
	local result_feat = torch.Tensor(dim[1],dim[2],1,dim[4],dim[5])
	for b=1,dim[1] do
		for i=1,length do
			for height=1,dim[4] do
				for width=1,dim[5] do
					result_feat[b][i][1][height][width] = torch.mean(fea[{{b},{i},{},{height},{width}}])      --mean pooling
					--result_feat[1][i][1][height][width] = torch.max(fea[{{1},{i},{},{height},{width}}])      --max pooling
				
				end
			end
		end
	end
	return result_feat
end

