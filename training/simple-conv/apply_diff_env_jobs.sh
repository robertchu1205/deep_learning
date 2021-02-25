for i in 96; do
  export IMG_WIDTH=$i
  export IMG_HEIGHT=$i
  export WH=wh-$i-dcgan
  for l in 3; do
    for f in 2 4 8 16; do
      export filters=$f
      export layers=$l
      # export maxbool=false
      # export nameSuffix=$i-$l$f-false
      # envsubst < /bigdata/robert/p3-saiap/nets/Simple_Conv/base/job.yaml | kubectl apply -f -
      export nameSuffix=$i-$l$f-true
      export maxbool=true
      envsubst < /bigdata/robert/p3-saiap/nets/Simple_Conv/base/job.yaml | kubectl apply -f -
    done
  done
done
