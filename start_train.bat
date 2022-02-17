@REM cd F:/_项目_/ai芯片/demo270/model

@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/lgb_model.py --valid --train --axis=x
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/lgb_model.py --valid --train --axis=y
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/error_analyze.py --model=lgb

@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/nn_model.py --model=dnn --valid --train --axis=x
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/nn_model.py --model=dnn --valid --train --axis=y
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/error_analyze.py --model=dnn

@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/nn_model.py --model=cnn --valid --axis=x
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/nn_model.py --model=cnn --valid --axis=y
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/error_analyze.py --model=cnn

F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/nn_model.py --model=lstm --valid --train --axis=x
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/model/nn_model.py --model=lstm --valid --train --axis=y
@REM F:\Anaconda3\envs\rl\python.exe F:/_项目_/ai芯片/demo270/error_analyze.py --model=lstm

pause