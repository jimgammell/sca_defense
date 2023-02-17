import numpy as np
from training.common import detach_result
from matplotlib import pyplot as plt
import os
import torch

animate_decision_boundary_object = None

def init_gan(save_dir):
    global animate_decision_boundary_object
    animate_decision_boundary_object = AnimateDecisionBoundaryOverTime(save_dir)

def get_decision_boundary_frame(disc, gen, dataset):
    global animate_decision_boundary_object
    animate_decision_boundary_object.save_plot(disc, gen, dataset)
    return True

def get_accuracy(_, labels, logits, **kwargs):
    logits = detach_result(logits)
    labels = detach_result(labels)
    predictions = np.argmax(logits, axis=-1)
    correctness = np.equal(predictions, labels)
    accuracy = np.mean(correctness)
    return accuracy

def get_mean_rank(_, labels, logits, **kwargs):
    logits = detach_result(logits)
    labels = detach_result(labels)
    ranks = np.array([np.count_nonzero(logits[idx]>=logits[idx][label])
                      for idx, label in enumerate(labels)])
    mean_rank = np.mean(ranks)
    return mean_rank

def get_confusion_matrix(_, labels, logits, **kwargs):
    logits = detach_result(logits)
    labels = detach_result(labels)
    confusion_matrix = np.zeros((logits.shape[-1], logits.shape[-1]))
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        confusion_matrix[prediction, label] += 1
    confusion_matrix /= logits.shape[0]/logits.shape[-1]
    return confusion_matrix

@torch.no_grad()
def get_generator_oracle_loss(disc, gen, dataset):
    [wi, wo] = [layer.weight for layer in next(gen.children()).children()]
    w = wo@wi
    neye = torch.zeros_like(w[:2, :2])
    neye[0, 0] = neye[1, 1] = -1.0
    oracle_loss = (w[:2, :2]-neye).norm()
    oracle_loss += w[2:, :2].norm()
    oracle_loss = float(oracle_loss.norm().cpu().numpy())
    return oracle_loss

class AnimateDecisionBoundaryOverTime:
    def __init__(self, save_dir, xlims=[-5, 5], ylims=[-5, 5]):
        self.save_dir = save_dir
        self.xlims = xlims
        self.ylims = ylims
        self.t = 0
    
    @torch.no_grad()
    def get_obfuscated_datapoints(self, dataset, gen):
        obscured_datapoints = torch.from_numpy(np.zeros_like(dataset.x)).to(torch.float)
        for idx, (x, _, _) in enumerate(dataset):
            obscured_datapoints[idx, :] = x+gen(x)
        if obscured_datapoints.shape[-1] != 2:
            obscured_datapoints = obscured_datapoints[:, :2]
        return obscured_datapoints
    
    @torch.no_grad()
    def get_class_means(self, dataset, gen):
        logits = gen(torch.tensor(dataset.feature_means).to(torch.float)).numpy()
        obscured_samples = dataset.feature_means + logits
        return obscured_samples
    
    #def get_decision_boundary(self, model):
    #    w = next(model.children()).weight.data.clone().detach().numpy()
    #    b = next(model.children()).bias.data.clone().detach().numpy()
    #    def x2(x1):
    #        return ((w[1, 0]-w[0, 0])/(w[0, 1]-w[1, 1]))*x1 + ((b[1]-b[0])/(w[0, 1]-w[1, 1]))
    #    direction = int(w[0, 1] >= w[1, 1])
    #    return x2, direction
    
    def plot_gan_scenario(self, disc, gen, dataset):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        obfuscated_datapoints = self.get_obfuscated_datapoints(dataset, gen)
        #decision_boundary, direction = self.get_decision_boundary(disc)
        ax.plot(obfuscated_datapoints[dataset.y==0][:, 0], obfuscated_datapoints[dataset.y==0][:, 1],
                '.', color='blue', label='Class 1', alpha=0.9)
        ax.plot(obfuscated_datapoints[dataset.y==1][:, 0], obfuscated_datapoints[dataset.y==1][:, 1],
                '.', color='red', label='Class 2', alpha=0.9)
        class_means = self.get_class_means(dataset, gen)
        
        res = 400
        xx, yy = np.meshgrid(np.linspace(*self.xlims, res), np.linspace(*self.ylims, res))
        with torch.no_grad():
            inp = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).to(torch.float)
            zz = disc(inp).numpy()
            zz = np.argmax(zz, axis=-1)
        zz = zz.reshape(xx.shape)
        ax.contourf(xx, yy, zz, colors=['blue', 'red'], alpha=0.5, levels=1)
        
        
        #xx = np.array(self.xlims)
        #ax.plot(xx, decision_boundary(xx), '--', color='purple', label='Decision boundary')
        #if direction == 0:
        #    ax.fill_between(xx, self.ylims[0], decision_boundary(xx), color='blue', alpha=0.5)
        #    ax.fill_between(xx, decision_boundary(xx), self.ylims[1], color='red', alpha=0.5)
        #elif direction == 1:
        #    ax.fill_between(xx, self.ylims[0], decision_boundary(xx), color='red', alpha=0.5)
        #    ax.fill_between(xx, decision_boundary(xx), self.ylims[1], color='blue', alpha=0.5)
        #ax.plot(class_means[0, 0], class_means[0, 1],
        #        marker='$1$', markersize=10, linestyle='none', color='black', markerfacecolor='none', label='Class 1 mean')
        #ax.plot(class_means[1, 0], class_means[1, 1],
        #        marker='$2$', markersize=10, linestyle='none', color='black', markerfacecolor='none', label='Class 2 mean')
        #ax.plot(class_means[:, 0], class_means[:, 1], '--', color='black')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Epoch %d'%(self.t))
        ax.set_xlim(*self.xlims)
        ax.set_ylim(*self.ylims)
        ax.legend(loc='lower right')
        return fig
    
    def save_plot(self, disc, gen, dataset):
        fig = self.plot_gan_scenario(disc, gen, dataset)
        fig.savefig(os.path.join('.', 'results', self.save_dir, 'frame_%d.png'%(self.t)), dpi=72)
        plt.close(fig)
        self.t += 1