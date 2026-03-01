-- 定义一个临时表 BlockFees，用于计算每个区块的总手续费
WITH BlockFees AS (
  SELECT
    block_number,
    SUM(fee) as total_fee_satoshi
  FROM
    `bigquery-public-data.crypto_bitcoin.transactions`
  WHERE
    block_timestamp >= '2023-01-01 00:00:00'
  GROUP BY
    block_number
),

-- 获取每个区块的 Coinbase 交易总输出
CoinbaseTx AS (
  SELECT
    block_number,
    SUM(output_value) as coinbase_output_satoshi
  FROM
    `bigquery-public-data.crypto_bitcoin.transactions`
  WHERE
    block_timestamp >= '2023-01-01 00:00:00'
    AND is_coinbase = TRUE
  GROUP BY
    block_number
)

SELECT
  b.number as height,
  b.timestamp,
  b.bits,
  -- 矿工总收入
  COALESCE(c.coinbase_output_satoshi, 0) AS miner_revenue_satoshi,
  -- 手续费
  COALESCE(f.total_fee_satoshi, 0) AS total_fee_satoshi
FROM
  `bigquery-public-data.crypto_bitcoin.blocks` b
LEFT JOIN
  BlockFees f ON b.number = f.block_number
LEFT JOIN
  CoinbaseTx c ON b.number = c.block_number
WHERE
  b.timestamp >= '2023-01-01 00:00:00'
ORDER BY
  b.number ASC;