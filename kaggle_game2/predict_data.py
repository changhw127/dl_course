import pandas as pd


def get_predict_result(s):
    emj = 0
    return emj


def run_predict():
    file = 'result/pred_data.csv'
    out_file = 'predict_result.csv'
    df = pd.DataFrame(pd.read_csv(file))
    for ix in df.index:
        s = df.loc[ix, 'Content']
        emj = get_predict_result(s)
        df.loc[ix, 'Label'] = emj
    df.to_csv('result/{}'.format(out_file), index=None)
    df = df[df['IntID','Label']]
    df.to_csv('pred/{}'.format(out_file), index=None)
    return True

if __name__ == '__main__':
    run_predict()
