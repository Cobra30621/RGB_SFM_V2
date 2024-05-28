import wandb
api = wandb.Api()
run = api.run("/tonytu/paper experiment/runs/qpi2aqvo")
print(run.config.keys())
run.config['group'] = '5/29'
run.config['description'] = "5/27 gauss + cReLU_percent(全域threshold) +SFM新合併方式 in MultiGrayDataset"
run.config.update()