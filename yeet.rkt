#lang racket

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

(foo `(3 2 1))
    