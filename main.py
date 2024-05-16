import pandas as pd
from dask import dataframe as dd
import wallets_analis_funcs as wa
from dask import delayed, compute

pd.set_option('display.expand_frame_repr', False)  # не переносить строки
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 20000)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.max_rows = 150

def main():
    
    DATA_FILE_DIR = 'I:\\Traider_bot\\парсинг транзакций\\parsing\\data'
    WALLETS_FILE_DIR = 'I:\\Traider_bot\\кошельки\\готовые данные по кошелькам\\данные всех кошельков спарсенных транзакций'

    WALLETS_IN_ITERATION = 100000
    FILES_IN_ITERATION = 3000
    ITERATIONS_TO_SAVE = 1
    WORKERS = 6
    MEMORY_LIMIT = '31GB'
    NPARTITIONS = 40
    PARTITION_SIZE_WALLETS = '300MB'
    PARTITION_SIZE = '500MB'
    THREADS_PER_WORKER = 2

    TEST_PROCESS = 0
    TEST_SAVE = 0
    
    if TEST_PROCESS:
        DATA_FILE_DIR = 'I:\\Traider_bot\\парсинг транзакций\\parsing\\data_test'
        WALLETS_IN_ITERATION = 10
        wa.make_unique_wallets_file_small()
        
    data_files, wallets_df, empty_pd_df, client = wa.init(MEMORY_LIMIT, THREADS_PER_WORKER, WORKERS, DATA_FILE_DIR, NPARTITIONS)
    ddf = dd.read_parquet(data_files, engine='fastparquet', columns=['Transaction_id', 'Wallet_id', 'Btc_block_time_price', 'Block_time', 'Amount', 'Transactions_Count'])
    ddf = ddf.repartition(partition_size=PARTITION_SIZE)
    txs_ddf = client.persist(ddf)
    
    batches = wa.get_unique_wallets(data_files, WALLETS_IN_ITERATION, NPARTITIONS, PARTITION_SIZE_WALLETS)
    iteration_counter = 0
    for i, wallets_batch in enumerate(batches):
        columns = [
                'Wallet_id', 'Avg_btc_diff_price', 'Total_amount', 'Total_txs_per_wallet', 'Avg_action_days',
                'Avg_btc_sell_price', 'Avg_btc_buy_price',
                'Sell_txs_quant', 'Buy_txs_quant', 
                'Sell_amount', 'Buy_amount',
                'Avg_sell_days', 'Avg_buy_days'
                    ]

        empty_pd_df = pd.DataFrame(columns=columns)
        wallets_df = dd.from_pandas(empty_pd_df, npartitions=NPARTITIONS)
        group_wallets_info = wa.process_wallet_group(wallets_batch, txs_ddf, FILES_IN_ITERATION, NPARTITIONS, PARTITION_SIZE, WORKERS)
        wallets_df = dd.concat([wallets_df, group_wallets_info])
        
        wallets_df = wallets_df.repartition(partition_size=PARTITION_SIZE)
        iteration_counter += 1
        if iteration_counter >= ITERATIONS_TO_SAVE and not TEST_SAVE:
            wa.save_wallets_tp_parquet(wallets_df, empty_pd_df, wallets_batch, client, i, NPARTITIONS, WALLETS_FILE_DIR)

if __name__ == "__main__":
    main()


    


