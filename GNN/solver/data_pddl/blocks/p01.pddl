;; blocks=5, out_folder=testing/easy, instance_id=1, seed=1007

(define (problem blocksworld-01)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 - object)
 (:init 
    (handempty)
    (clear b3)
    (on b3 b5)
    (on b5 b4)
    (ontable b4)
    (clear b2)
    (on b2 b1)
    (ontable b1))
 (:goal  (and 
    (clear b4)
    (on b4 b3)
    (ontable b3)
    (clear b2)
    (ontable b2)
    (clear b1)
    (on b1 b5)
    (ontable b5))))
