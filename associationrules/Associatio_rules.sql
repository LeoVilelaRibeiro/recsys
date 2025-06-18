with cte_item_orders as (
  select
    order_gold._p_retailer,
    order_gold._id,
    order_gold.orderGwId,
    order_gold._p_customer,
    order_gold.status,
    explode(from_json(items, 'array<string>')) as item
  from
    mongodb.order_gold
),
cte_order as (
  select
    _p_retailer,
    _id,
    orderGwId,
    _p_customer,
    status,
    collect_list(item_id) as items
  from (
    select
      _p_retailer,
      _id,
      orderGwId,
      _p_customer,
      status,
      from_json(item, '__type string, className string, objectId string').objectId as item_id
    from cte_item_orders
  )
  group by
    _p_retailer,
    _id,
    orderGwId,
    status,
    _p_customer
)

select 
  cte_order.`_p_retailer`,
  --customer_gold.name,
  --cte_order.`_id` as order_id_interno,
  --cte_order.orderGwId as order_id_platform,
  --cte_order.status,
  items
from cte_order
inner join mongodb.customer_gold where customer_gold._id = cte_order.`_p_customer`
order by cte_order.`_p_retailer`;
--having cte_order.`_p_retailer`= '0GCBrJGu9f';