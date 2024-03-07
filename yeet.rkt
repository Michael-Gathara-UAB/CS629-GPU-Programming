#lang racket
(require racket/trace)

(define (foo ls)
    (define v (baz ls))
    (let ([lst (boop ls v ls)])
        lst))

(define (boop r i d)
    (if (> i 0)
        (cons (car r)
            (boop (cdr r) (- i 1) d))
        d))

(define (baz q)
    (let ([q (cdr q)])
        (car q)))

(trace foo)
(trace boop)
(trace baz)
(foo `(3 2 1)) ; prints out `(3 2 3 2 1)
    