from training.common import to_np, accuracy, local_avg

def train_step(batch, model, loss_fn, optimizer, device, grad_clip=None):
    trace, label = batch
    trace = trace.to(device)
    label = label.to(device)
    model.train()
    logits = model(trace)
    loss = loss_fn(logits, label)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip != None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
    optimizer.step()
    return {'loss': from_np(loss), 'acc': accuracy(logits, label)}