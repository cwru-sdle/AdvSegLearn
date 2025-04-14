import torch
from matplotlib import pyplot as plt
import numpy 
import random
import pandas as pd
import time
import os
class train_GAN:
    def __init__(self,
    disc_loss,
    seg_gen_loss,
    unseg_gen_loss,
    seg_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    seg_train_dl: torch.utils.data.DataLoader,
    seg_val_dl: torch.utils.data.DataLoader,
    unseg_train_dl: torch.utils.data.DataLoader,
    epochs:int,
    gen_optimizer:torch.optim.Optimizer,
    disc_optimizer:torch.optim.Optimizer,
    order = [[0,1],[2,3,4]],
    accumulation_steps:int = 1,
    timer = True,
    device = None
    ):
        self.disc_loss = disc_loss
        self.seg_gen_loss = seg_gen_loss
        self.unseg_gen_loss = unseg_gen_loss
        self.seg_model = seg_model
        self.disc_model = disc_model
        self.seg_train_dl = seg_train_dl
        self.seg_val_dl = seg_val_dl
        self.unseg_train_dl = unseg_train_dl
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.order = order
        self.timer = timer
        self.device = device
    @staticmethod
    def metrics_list(input:list[float]):
        if(type(input)!=list):
            raise ValueError('Input must be a list. was a: '+str(type(input)))
        return (
            numpy.mean(input),
            numpy.percentile(input,25),
            numpy.percentile(input,50),
            numpy.percentile(input,75)
        )
    @staticmethod
    def save_loss_plot(
                    loss_metrics:list[list[tuple[float]]],
                    legend_titles:list[str],
                    subplot_titles:list[str],
                    order:list[list[int]],
                    save_path_fig,
                    save_path_csv
    ):
        # The loss metric should have 3 dimensions: [metric,epoch,quartile], which is then converted to: [metric,quartile,epoch]
        # The order should have 2 dimensions: [suplot, metric]
        fig, ax = plt.subplots(len(order),figsize=(14,10))
        if len(order)==1:
            ax=[ax]
        series_list = []
        for metric in range(len(loss_metrics)):
            loss_metrics[metric] = list(zip(*loss_metrics[metric]))
            for i, quartile in enumerate(loss_metrics[metric]):
                name = legend_titles[metric]
                if i==0:
                    name+='_mean'
                elif i==1:
                    name+='_q1'
                elif i==2:
                    name+='_q2'
                elif i==3:
                    name+='_q3'
                else:
                    raise ValueError('There are more than 4 quartiles points in the distribution of the metric being measured.')
                series_list.append(
                    pd.Series(quartile,name = name)
                )
        csv_df = pd.DataFrame(series_list)
        csv_df.to_csv(save_path_csv)
        for subplot, list_inner in enumerate(order):
            temp_legend_titles = []
            for position, metric in enumerate(list_inner):
                if metric == 0:
                    color_str = 'r'
                elif metric == 1:
                    color_str = 'b'
                elif metric ==2:
                    color_str='g'
                else:
                    color_str = (random.random(),random.random(),random.random())
                epochs = len(loss_metrics[metric][0])
                ax[subplot].plot(range(1,epochs+1),loss_metrics[metric][0],ls='-',color=color_str,lw=0.5) #Mean
                ax[subplot].plot(range(1,epochs+1),loss_metrics[metric][2],ls='--',color=color_str,lw=0.5) #Median
                ax[subplot].fill_between(
                    range(1,epochs+1),
                    loss_metrics[metric][1],#q1
                    loss_metrics[metric][3],#q3
                    color=color_str,
                    alpha = 0.4,
                    lw=0.5
                )
                temp_legend_titles.append(legend_titles[order[subplot][position]]+' mean') 
                temp_legend_titles.append(legend_titles[order[subplot][position]]+' median') 
                temp_legend_titles.append(legend_titles[order[subplot][position]]+'IQR') 
                ax[subplot].set_ylabel(subplot_titles[subplot])
            ax[subplot].legend(temp_legend_titles,fontsize='large')
            ax[subplot].set_xlabel('Epochs')
            ax[subplot].set_ylim(bottom=0,top=2*(sum(loss_metrics[metric][0])/len(loss_metrics[metric][0])))
        fig.tight_layout()
        fig.savefig(save_path_fig)
        plt.close(fig)
    def run_optimizer(self,idx, optimizer: torch.optim.Optimizer):
        if (idx+1)%self.accumulation_steps==0:
            optimizer.step()
            optimizer.zero_grad()
    def save_script(path:str):
        # Get the path of the current script
        script_path = os.path.abspath(__file__)

        # Open the script itself and read its contents
        with open(script_path, 'r') as script_file:
            script_content = script_file.read()

        # Define the path where you want to save the log (e.g., folder 'logs')

        # Write the content of the script into the log file
        with open(path, 'w') as log_file:
            log_file.write(script_content)

        print(f"Script content has been logged to {path}")
    def up_seg_unsup(self):
        model_output = self.seg_model(self.raw)
        disc_output = self.disc_model(model_output).detach()
        #up the segmentation model
        unsup_seg_loss_temp_point = self.unseg_gen_loss(disc_output, model_output)
        unsup_seg_loss_temp_point.backward()
        self.unsup_seg_loss_temp.append(unsup_seg_loss_temp_point.item())
        self.run_optimizer(self.idx,self.gen_optimizer)
    def up_seg_model(self):
        sup_seg_loss_temp_point = self.seg_gen_loss(self.model_output,self.true,self.seg_disc_output.detach())
        sup_seg_loss_temp_point.backward()
        self.run_optimizer(self.idx,self.gen_optimizer)
        self.sup_seg_loss_temp.append(sup_seg_loss_temp_point.item())
    def up_disc_model(self):
        #on segmentation model
        seg_disc_loss_temp_point = self.disc_loss(self.seg_disc_output,False)
        self.seg_disc_loss_temp.append(seg_disc_loss_temp_point.item())
        #on raw data
        raw_disc_loss_temp_point= self.disc_loss(self.raw_disc_output,True)
        self.raw_disc_loss_temp.append(raw_disc_loss_temp_point.item())

        temp = seg_disc_loss_temp_point + raw_disc_loss_temp_point
        temp.backward()
        self.run_optimizer(self.idx,self.disc_optimizer)
    def segmented_training_epoch(self):
        self.sup_seg_loss_temp= []
        self.seg_disc_loss_temp = []
        self.raw_disc_loss_temp = []
        for idx, datapoint in enumerate(self.seg_train_dl):
            self.idx=idx
            if not self.device is None:
                self.raw = datapoint[0].to(self.device)
                self.true = datapoint[1].to(self.device)
            else:
                self.raw = datapoint[0]
                self.true = datapoint[1]
            self.model_output = self.seg_model(self.raw)
            self.seg_disc_output = self.disc_model(self.model_output.detach())
            self.raw_disc_output = self.disc_model(self.true)
            self.up_seg_model()
            self.up_disc_model()
        self.sup_seg_loss.append(self.metrics_list(self.sup_seg_loss_temp))
        self.seg_disc_loss.append(self.metrics_list(self.seg_disc_loss_temp))
        self.raw_disc_loss.append(self.metrics_list(self.raw_disc_loss_temp))
    def unsegmented_training_epoch(self):
        self.unsup_seg_loss_temp = []
        for idx, datapoint in enumerate(self.unseg_train_dl):
            if not self.device is None:
                self.raw = datapoint.to(self.device)
            else:
                self.raw = datapoint
            self.up_seg_unsup()
        self.unsup_seg_loss.append(self.metrics_list(self.unsup_seg_loss_temp))

    def validation_epoch(self):
        val_seg_loss_temp = []
        for idx, datapoint in enumerate(self.seg_val_dl):
            if not self.device is None:
                raw = datapoint[0].to(self.device)
                true = datapoint[1].to(self.device)
            else:
                raw = datapoint[0]
                true = datapoint[1]
            model_output = self.seg_model(raw)
            #Save the validation loss 
            disc_output = self.disc_model(model_output)
            val_seg_loss_temp_point = self.seg_gen_loss(model_output,true,disc_output)
            val_seg_loss_temp.append(val_seg_loss_temp_point.item())
        self.val_seg_loss.append(self.metrics_list(val_seg_loss_temp))


    def adversarial_learning(self,save_path=None):
        self.raw_disc_loss = []
        self.seg_disc_loss = []
        self.sup_seg_loss = []
        self.unsup_seg_loss = []
        self.val_seg_loss = []
        if(not self.timer):
            for epoch in range(self.epochs):
                #Supervised
                self.segmented_training_epoch()

                #Unsupervised
                self.unsegmented_training_epoch()
                
                #Validation epoch
                self.validation_epoch()
        else:
            start_time = time.time()
            print('Time: '+str(time.time()-start_time))
            for epoch in range(self.epochs):
                #Supervised
                print(f'start seg training epoch {epoch}: '+str(time.time()-start_time))
                self.segmented_training_epoch()

                #Unsupervised
                print(f'start unseg training epoch {epoch}: '+str(time.time()-start_time))
                self.unsegmented_training_epoch()
                
                #Validation epoch
                print(f'start validation epoch {epoch}: '+str(time.time()-start_time))
                self.validation_epoch()
        if save_path!=None:
            self.save_loss_plot(
                        loss_metrics=[
                            self.raw_disc_loss,
                            self.seg_disc_loss,
                            self.sup_seg_loss,
                            self.unsup_seg_loss,
                            self.val_seg_loss,
                        ],
                        legend_titles = [
                            'raw_disc_loss',
                            'seg_disc_loss',
                            'sup_seg_loss',
                            'unsup_seg_loss',
                            "val_seg_loss",
                        ],
                        subplot_titles = [
                            'BCE training loss',
                            'BCE validation loss'
                        ],
                        order=self.order,
                        save_path_fig=save_path+'/figure.png',
                        save_path_csv=save_path+'/figure,csv'
            )
            torch.save(self.seg_model.state_dict(),save_path+'/model.pt')
            self.save_script(save_path+'/train_GAN_script.txt')