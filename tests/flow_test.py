# import flow.flowequation as feq
import flow.twophase as tf

# only works if the command python -m tests.flow_test is used
# example problem in Two Phase Flow in Pipes by Beggs / Brill (1988) pg. 3-31
book_Ngv = 9.29
book_Nlv = 6.02
book_Nd = 41.34
book_l1, book_l2 = 1.53, 0.88
book_vpat = "slug"

calc_l1, calc_l2 = tf.ros_lp(book_Nd)
calc_vpat = tf.ros_flow_pattern(9.29, 6.02, 41.34)

print(f"Ros L1 & L2 - Book: {book_l1} {book_l2}, Calc: {round(calc_l1, 2)} {round(calc_l2, 2)}")
print(f"Ros Regime - Book: {book_vpat}, Calc: {calc_vpat}")

# example problem in Two Phase Flow in Pipes by Beggs / Brill (1988) pg. 3-62
book_nslh = 0.393
book_NFr = 5.67
book_hpat = "intermittent"
book_tparm = 1  # this is really n/a since it is intermittent flow
book_ilh = 0.512

calc_hpat, calc_tparm = tf.beggs_flow_pattern(book_nslh, book_NFr)
calc_ilh = tf.beggs_holdup_inc(0.393, 5.67, 6.02, 90, calc_hpat, calc_tparm)

print(f"Beggs Regime - Book: {book_hpat}, Calc: {calc_hpat}")
print(f"Beggs Incline Liquid Holdup - Book: {book_ilh}, Calc: {round(calc_ilh, 3)}")
