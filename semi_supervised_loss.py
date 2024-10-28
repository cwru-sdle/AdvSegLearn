import torch

class semi_supervised_loss(torch.nn.Module):

    def __init__(self, threshold:float = .3, mutually_exclusive: list[set[int]] = []):
        super(semi_supervised_loss,self).__init__()
        self.threshold = threshold
        self.mutually_exclusive = mutually_exclusive

    def forward(
        self,
        discriminator_output: torch.Tensor,
        segmentation_output: torch.Tensor,
    ):
        disc_temp = self.indicator_function(discriminator_output)
        seg_temp, position_indexes = self.seperate_tensor(segmentation_output) #positions = list[list[int]], seg_temp=list[torch.Tensor]
        for i, exclusive_positions in enumerate(position_indexes):
            if(len(exclusive_positions)==1):
                seg_temp[i] = torch.unsqueeze(seg_temp[i][0],-3)
            else:
                seg_temp[i] = torch.stack(seg_temp[i],dim=-3)
            seg_temp[i]= self.one_hot_encoder_tensor_2d_batched(seg_temp[i]) #only classes predications above the threshold can be one-hot encoded
        
        flattened_tensor_list = [None]*segmentation_output.shape[-3]
        for i, position_set in enumerate(position_indexes):# i will index through seg_temp, position_set is used tomake the next list
            for j, position in enumerate(position_set): # j will index through each tensor, position will index trhough the flattened_tensor_list
                flattened_tensor_list[position] = torch.select(seg_temp[i],-3,j)
        one_hot_encoded_segmentation_output = torch.stack(flattened_tensor_list,dim=-3)
        log_seg_output = torch.log(segmentation_output+0.00001)
        return -torch.mean(disc_temp*one_hot_encoded_segmentation_output*log_seg_output)

    def seperate_tensor(self,input:torch.Tensor):
        tensor_list = []
        position = []
        moved = [False]*input.shape[-3]
        #Seperating the channels in the mutually exclusive list
        for i in self.mutually_exclusive:
            temp_tensor_list = []
            temp_position = []
            for j in i:
                temp_tensor_list.append(
                            torch.select(input,-3,j)
                )
                temp_position.append(j)
                moved[j]=True
            tensor_list.append(temp_tensor_list)
            position.append(temp_position)
        #Seperating the remaining channels
        for i, value in enumerate(moved):
            if(not value):
                tensor_list.append(
                    [
                        torch.select(input,-3,i)
                    ]
                )
                position.append([i])
        return tensor_list, position

    def one_hot_encoder_tensor_2d_batched(self,input):
        #Expects 1 or more channels
        try:
            BATCH_SIZE = input.shape[0]
            CHANNELS = input.shape[1]
            X_SIZE = input.shape[2]
            Y_SIZE = input.shape[3]
        except IndexError:
            print('The tensor is not the appropriate shape.')
            print(input.shape)
        threshold_tensor = torch.zeros(BATCH_SIZE,1,input.shape[-2],input.shape[-1])+self.threshold #no class channel made
        threshold_tensor = threshold_tensor.to(input.device)
        x = torch.concat([threshold_tensor,input],dim=-3) #Add no class to input
        x = torch.argmax(x,dim=-3,keepdim=False) #Find largest value
        x = torch.nn.functional.one_hot(x,CHANNELS+1) #Convert to one hot encoded tensor
        x = torch.permute(x,(0,-1,1,2)) #Convert onehot-encoded to the appropriate shape
        x = x[:,1:CHANNELS+1,:,:] #Remove no class channel
        return x

    def indicator_function(self,input):
        return torch.sigmoid((input-self.threshold)*1000)