import torch

def check_mutually_exclusive(mutually_exclusive, input_tensor_shape):
    if mutually_exclusive:
        if  set.intersection(*mutually_exclusive) != set():
            raise ValueError(f'Cannot softmax if sets of mutually exclusive features have the same element. In this list of mutually exclusive sets, the element {set.intersection(*mutually_exclusive)} was in two or more sets.'
            )
        if max(set.union(*mutually_exclusive)) >= input_tensor_shape[-3] or min(set.union(*mutually_exclusive)) < 0:
            raise ValueError(
                f'The largest index contained is {max(set.union(*mutually_exclusive))} and the smallest index is {min(set.union(*mutually_exclusive))} while the shape of the segmented tensor is {input_tensor_shape}. Indices cannot be outside of the range of image channels.'
            )

def slice_and_softmax(input_tensor, sets, mutually_exclusive_loss, temp, set_index, tracker):
    temp.append([])
    for channel in sets:
        temp[set_index].append(
            torch.select(
                input_tensor,
                -3,
                channel
            )
        )
        tracker[channel]=True
    temp[set_index] = torch.stack(temp[set_index], -3)
    # Applies softmax to those channels
    for batch in range(input_tensor.shape[-4]):
        for y_cord in range(input_tensor.shape[-2]):
            for x_cord in range(input_tensor.shape[-1]):
                tensor_slices = []
                for index in range(len(sets)):
                    tensor_slices.append(
                        temp[set_index][batch, index, y_cord, x_cord]
                    )
                tensor_slices = mutually_exclusive_loss(torch.Tensor(tensor_slices).float())
                for index in range(len(sets)):
                    temp[set_index][batch, index, y_cord, x_cord] = tensor_slices[index]
    return temp

def apply_standard_activation(input_tensor, standard_loss, tracker, channels):
    for tracker_index, channel in enumerate(tracker):
        if not channel:
            channels[tracker_index] = standard_loss(
                torch.select(
                    input_tensor,
                    dim=-3,
                    index=tracker_index
                )
            )
    return channels

def mixed_activation(
    input_tensor: torch.Tensor,
    mutually_exclusive_loss = torch.nn.Softmax(dim=0),
    standard_loss = torch.tanh,
    mutually_exclusive: list[set[int]] = []
):
    check_mutually_exclusive(mutually_exclusive, input_tensor.shape)
    tensor_slices = []
    tracker = [False] * input_tensor.shape[-3]
    channels = [None] * input_tensor.shape[-3]
    temp = []
    for set_index, sets in enumerate(mutually_exclusive):
        temp = slice_and_softmax(input_tensor=input_tensor, sets=sets, mutually_exclusive_loss=mutually_exclusive_loss, temp=temp, set_index=set_index, tracker=tracker)
        # Seperate the previously merged channels
        for i, channel in enumerate(sets):
            channels[channel] = torch.select(temp[set_index], dim=-3, index=i)
        
    channels = apply_standard_activation(input_tensor, standard_loss, tracker, channels)
    
    # stacks into original diameters
    output = torch.stack(channels, dim=-3)
    del temp, tensor_slices
    return output
